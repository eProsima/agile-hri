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

import rclpy
import re
import torch

from TTS.api import TTS

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from hri_msgs.action import Tts  # Import TTSPhrase message type

# Dictionary of phones for the letters in the English alphabet
initials_phones = {
    "A": "ay",
    "B": "bee",
    "C": "sieh",
    "D": "dee",
    "E": "ee",
    "F": "eff",
    "G": "jie",
    "H": "edge",
    "I": "eye",
    "J": "jay",
    "K": "kaye",
    "L": "elle",
    "M": "emme",
    "N": "en",
    "O": "owe",
    "P": "pea",
    "Q": "queue",
    "R": "are",
    "S": "ess",
    "T": "tea",
    "U": "hugh",
    "V": "vee",
    "W": "doub you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}


def preprocess_phrase(phrase):
    """
    Replaces any acronyms in the input phrase with their corresponding phonetic representations,
    appending a comma after each replacement.

    :param phrase: The input phrase to preprocess.
    :return: The preprocessed phrase with acronyms replaced by their phonetic representations followed by a comma.
    """
    # Function to replace acronym with its phonetic representation
    def replace_acronym(acronym):
        return ' '.join(initials_phones[letter] for letter in acronym)

    # Split the input phrase into words
    words = phrase.split()
    processed_words = []

    for word in words:
        if re.search(r'[A-Z]{2,}', word):  # Check if the word contains an acronym
            acronym = re.findall(r'[A-Z]+', word)[0]  # Extract the acronym part
            phonetic_acronym = replace_acronym(acronym)
            processed_words.append(f"{phonetic_acronym}:")  # Append a comma after the phonetic representation
        else:
            processed_words.append(word)

    # Join the processed words into a single string and return it
    return ' '.join(processed_words)


class TTSGen(Node):
    """
    Node that offers a TTS action server and saves the output into the file specified (.wav).
    The purpose of this node is to generate audio files from text input.
    It can be run independently of the TTS publisher and subscriber nodes.
    """

    def __init__(self):
        super().__init__('hri_tts_gen')

        self.declare_parameter(
            'file_path', '/tmp/audio_generated.wav', ParameterDescriptor(
                description='Path to the audio file to be saved.'))
        self.file_path = self.get_parameter('file_path').value

        self._action_server = ActionServer(
            self,
            Tts,
            '/hri_tts',
            self.tts_server_callback,
            goal_callback=self.tts_goal_callback)

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {device}")
        # Init TTS with the target model name
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True).to(device)

        self.get_logger().info('TTS generator started')

    def tts_goal_callback(self, goal_request):
        """ Callback to accept or reject the TTS action. """
        if goal_request.speech is not None:
            self.get_logger().debug('Received request to start TTS.')
            return GoalResponse.ACCEPT
        self.get_logger().warn('Received wrong request to start TTS.')
        return GoalResponse.REJECT

    def tts_server_callback(self, goal_handle):
        """ Callback of the main TTS action server. """
        self.get_logger().info(f"Received speech: '{goal_handle.request.speech}'")
        # Replace any acronym with their corresponding phones
        phrase_preprocessed = preprocess_phrase(goal_handle.request.speech)

        try:
            # Call the TTS function to save the phrase to an audio file
            self.tts.tts_to_file(text=phrase_preprocessed,
                                    file_path=self.file_path,
                                    split_sentences=False)
            self.get_logger().info(f'Saved audio to {self.file_path}')
        except Exception as e:
            self.get_logger().error(f'Error generating audio file: {e}')
            goal_handle.abort()
            return

        goal_handle.succeed()
        result = Tts.Result()
        result.finished = True
        self.get_logger().debug('Goal succeeded.')

        return result


def main(args=None):
    rclpy.init(args=args)

    tts_pub = TTSGen()
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
        print("TTS generator node shutdown")
        # Destroy the node explicitly
        tts_pub.destroy_node()


if __name__ == '__main__':
    main()
