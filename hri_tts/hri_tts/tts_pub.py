# Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import rclpy
import re
import subprocess
import tempfile
import threading
import torch
import wave

from TTS.api import TTS

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool

from hri_msgs.action import Tts
from hri_msgs.msg import Audio
from builtin_interfaces.msg import Time
from hri_tts.tts_gen import preprocess_phrase

# Dictionary of preset phrases where a string matches the number of the audio file to play
# 100: A little phrase for a man, but a big phrase for a robot.
# 101: Hello, I am your personal robot, how can I help you?
# 102: Goodbye.
# 103: I have finished the action you have requested, what else can I do for you?
# 104: I could not understand you.
# 105: *music*

preset_phrases = {
    "_example": 100,
    "_hello": 101,
    "_bye": 102,
    "_finish": 103,
    "_stt_error": 104,
    "_music": 105,
}

class TTSPub(Node):
    """
    Node that offers a TTS action server and publish the output into an Audio topic (raw bytes + metadata).
    The purpose of this node is to publish audio messages from text input.
    It requires to run a "playback" node capable of receiving and playing the audio messages to listen to them.
    """

    def __init__(self):
        super().__init__('hri_tts_pub')

        self.declare_parameter(
            'model', 'tts_models/en/ljspeech/tacotron2-DDC', ParameterDescriptor(
                description='TTS model name'))
        self.declare_parameter(
            'wait_for_finished', True, ParameterDescriptor(
                description='If the action should wait for the /hri_tts/finished message before completing (wait for audio playback).'))

        self.model_name = self.get_parameter('model').value
        self.wait_for_finished = self.get_parameter('wait_for_finished').value

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {device}")
        # Init TTS with the target model name
        self.tts = TTS(model_name=self.model_name, progress_bar=True).to(device)

        # Track if TTS finished
        self.condition = threading.Condition()   # Condition to wait for the /hri_tts/finished message
        self.tts_finished = False                # Flag to track when the TTS task is complete

        # QoS Configuration
        qos_1_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Create a publisher for the /hri_tts/audio topic
        self.audio_publisher = self.create_publisher(Audio, '/hri_tts/audio', qos_1_reliable)
        # Subscribe to the /hri_tts/finished topic
        self.finished_subscription = self.create_subscription(Bool, '/hri_tts/finished', self.finished_callback, qos_1_reliable)

        self._action_server = ActionServer(
            self,
            Tts,
            '/hri_tts',
            self.tts_server_callback,
            goal_callback=self.tts_goal_callback,)

        self.get_logger().info('TTS node started')

    def tts_goal_callback(self, goal_request):
        """ Callback to accept or reject the TTS action. """
        if goal_request.speech is not None:
            self.get_logger().debug('Received request to start TTS.')
            return GoalResponse.ACCEPT
        self.get_logger().warn('Received wrong request to start TTS.')
        return GoalResponse.REJECT

    def tts_cancel_callback(self, goal_handle):
        """ Callback to cancel the TTS action. """
        with self.condition:
            self.get_logger().info('Received request to cancel in progress TTS.')
            self.tts_finished = True
            self.condition.notify_all()
        return CancelResponse.ACCEPT

    def tts_server_callback(self, goal_handle):
        """ Callback of the main TTS action server. """
        self.get_logger().debug('Executing goal...')

        # Reset the finished flag at the start of each goal
        if self.wait_for_finished:
            self.tts_finished = False
        else:
            self.tts_finished = True
        timeout_margin = 0

        # Initialize the feedback message
        feedback_msg = Tts.Feedback()

        # Initialize the audio message
        msg = Audio()
        msg.sample_rate = 44100
        msg.num_channels = 1
        msg.subtype = 'wav'

        # Check if the goal is a preset phrase
        preset_phrase_number = preset_phrases.get(goal_handle.request.speech, 0)
        self.get_logger().info(f"Received speech: '{goal_handle.request.speech}'")
        if preset_phrase_number > 0:
            # Publish the preset phrase number
            msg.preset = preset_phrase_number
            self.audio_publisher.publish(msg)

            # Send the feedback
            feedback_msg.sentences = 1
            feedback_msg.duration.sec = 20  # This should be adjusted to the highest preset saved
            feedback_msg.duration.nanosec = 0
            goal_handle.publish_feedback(feedback_msg)
            timeout_margin = feedback_msg.duration.sec

            self.get_logger().debug(f"Duration of {preset_phrase_number}: {feedback_msg.duration} seconds")

        # Handle text-to-speech synthesis for arbitrary phrases
        else:
            if goal_handle.request.speech == "":
                self.get_logger().warn("Received empty speech request.")
                result.finished = False
                goal_handle.abort()
                return
            # Call the function to process the incoming message
            phrases = self.split_into_phrases(goal_handle.request.speech)

            # Publish the number of phrases to be synthesized
            msg.sentences = len(phrases)
            msg.id = 0
            # Loop through each phrase and create an audio file
            for i, phrase in enumerate(phrases, start=1):
                # Replace any acronym with their corresponding phones
                phrase_preprocessed = preprocess_phrase(phrase)

                # Call the TTS function to save the phrase to an audio file
                temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=f'.wav')
                self.tts.tts_to_file(text=phrase_preprocessed,
                                     file_path=temp_file.name,
                                     split_sentences=False)
                self.get_logger().debug(f'Saved audio for preprocessed phrase "{phrase_preprocessed}" to {temp_file.name}')

                # Send the feedback
                feedback_msg.sentences += 1
                new_duration = self.get_wav_duration(temp_file.name)
                if new_duration is not None:
                    feedback_msg.duration = self.add_and_normalize(feedback_msg.duration, new_duration)
                goal_handle.publish_feedback(feedback_msg)
                timeout_margin += new_duration.sec
                self.get_logger().debug(f"Duration of {temp_file.name}: {feedback_msg.duration} seconds")

                # Send the audio file as bytes
                try:
                    audio_data = self.load_audio_bytes(temp_file.name)
                    msg.data = audio_data
                    msg.id = i
                    self.audio_publisher.publish(msg)
                    self.get_logger().info(f'Sent audio file: {temp_file.name} of {len(audio_data)} bytes')
                except Exception as e:
                    self.get_logger().warn(f'Failed to send file: {temp_file.name}: {e}')
                finally:
                    temp_file.close()

        # Now wait for the /hri_tts/finished message
        result = Tts.Result()
        with self.condition:
            if not (self.condition.wait_for(lambda: self.tts_finished, timeout=5+timeout_margin)):
                self.get_logger().warn('Aborting action due to timeout. Server did not receive /hri_tts/finished message.')
                goal_handle.abort()
                result.finished = False
                self.get_logger().warn('Goal failed.')
            else:
                goal_handle.succeed()
                result = Tts.Result()
                result.finished = True
                self.get_logger().debug('Goal succeeded.')

        return result

    def load_audio_bytes(self, file_path: str):
        """ Loads the audio file as bytes from the given file path. """
        with open(file_path, 'rb') as f:
            return f.read()

    def finished_callback(self, msg):
        """
        Callback for /hri_tts/finished topic, setting the flag to True.
        This topic is used to finish the action when the subscriber notifies that is has played the audio.
        """
        if msg.data:
            with self.condition:
                self.tts_finished = True  # Mark as finished
                self.get_logger().info('Message of /hri_tts/finished received')
                self.condition.notify_all()

    def split_into_phrases(self, message):
        """
        Splits the input message into a list of phrases separated by periods ('.').

        :param message: The input string message to split.
        :return: A list of phrases.
        """
        # First check if the message ends with a period, if not add it
        if not message.endswith('.'):
            message += '.'

        # Strip any extra whitespace and split by '.'
        phrases = [phrase.strip() for phrase in message.split('.') if phrase.strip()]
        # Add a period to the end of each phrase
        phrases = [f"{phrase}." for phrase in phrases]
        return phrases

    def get_wav_duration(self, file_path):
        """
        Gets the duration of a .wav file

        :param file_path: The path to the .wav file.
        :return: The duration of the .wav file
        """
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                time_msg = Time()

                # Split the duration into seconds and nanoseconds
                time_msg.sec = math.floor(duration)  # Integer part for seconds
                time_msg.nanosec = int((duration - time_msg.sec) * 1e9)  # Fractional part for nanoseconds

                return time_msg
        except FileNotFoundError:
            self.get_logger().error(f"File {file_path} not found.")
            return None

    def add_and_normalize(self, dst, delta):
        """
        Add delta (Time/Duration) into dst (Time/Duration) and normalize.
        Assumes fields: .sec (int), .nanosec (int, 0..1e9-1)

        :param dst: Destination Time/Duration to add into.
        :param delta: Time/Duration to add.
        :return: The normalized sum of dst and delta.
        """
        dst.sec += int(delta.sec)
        dst.nanosec += int(delta.nanosec)

        # Carry if nanosec overflowed
        if dst.nanosec >= 1_000_000_000:
            carry, dst.nanosec = divmod(dst.nanosec, 1_000_000_000)
            dst.sec += carry

        return dst


def main(args=None):
    rclpy.init(args=args)

    tts_pub = TTSPub()
    executor = MultiThreadedExecutor()
    executor.add_node(tts_pub)

    try:
        # Loop and call rclpy.spin_once repeatedly to handle callbacks
        executor.spin()
    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("Shutting down node (Ctrl+C pressed).")
    finally:
        # rclpy.shutdown() is already being called in the destructor of the Node
        print("TTS publisher node shutdown")
        # Destroy the node explicitly
        tts_pub.destroy_node()


if __name__ == '__main__':
    main()
