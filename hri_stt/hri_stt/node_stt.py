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


from hri_msgs.action import Stt
from hri_msgs.msg import FaceInterface
from hri_stt.tuning import Tuning
from rcl_interfaces.msg import ParameterDescriptor

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from faster_whisper import WhisperModel
from math import gcd
from scipy.signal import resample_poly
import numpy as np
import pyaudio
import tempfile
import threading
import time
import torch
import usb.core
import wave

# PyAudio Parameters
WIDTH = 2                                  # Audio format
SILENCE_BEFORE_SENDING = 1                 # Silence period in seconds before sending audio to start partial transcription
MAX_SILENCE_DURATION_SECONDS = 1.5         # Max silence duration in seconds before ending
MAX_INITIAL_SILENCE_DURATION_SECONDS = 8   # Max silence duration before listening any voice


# Provided by Alexander Veysov
def int2float(sound):
    """ Convert the audio chunk from int16 to float32 and normalize it. """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    # sound = sound.squeeze()  # depends on the use case
    return sound


def get_mic_index(microphone: str) -> int | None:
    """ Get the index of the microphone. """
    p = pyaudio.PyAudio()

    index = None
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if microphone in device_info["name"]:
            index = i
    p.terminate()
    return index

def get_cfg_mic(index: int) -> tuple[int, int] | tuple[None, None]:
    """ Get the configuration of the microphone. """
    p = pyaudio.PyAudio()
    try:
        device_info = p.get_device_info_by_index(index)
        channels = int(device_info['maxInputChannels'])
        rate = int(device_info['defaultSampleRate'])
        p.terminate()
        return channels, rate
    except:
        p.terminate()
    return None, None

def get_chunk_mic(rate: int, target_ms: float = 64.0, align: int = 256) -> int:
    """Pick frames_per_buffer so that each chunk ~ target_ms long; Use multiples of 256 with 'align'."""
    frames = int(round(rate * target_ms / 1000.0))
    if align:
        q = max(1, int(round(frames / align)))
        frames = q * align
    return frames


class NodeSTT(Node):
    def __init__(self):
        # Initialize node
        super().__init__('hri_stt')

        self.get_logger().info("Starting STT Node")

        self.declare_parameter(
            'microphone', '', ParameterDescriptor(
                description='Microphone name used by PyAudio to find the microphone device in the system.'))
        self.declare_parameter(
            'vad', 'silero', ParameterDescriptor(
                description='VAD selection. Options: silero, mic'))
        self.declare_parameter(
            'config_file', 'noise_config.json', ParameterDescriptor(
                description='Microphone configuration file'))
        self.declare_parameter(
            'whisper_model', 'medium.en', ParameterDescriptor(
                description='Whisper model selection.'))
        self.declare_parameter(
            'gpu_load', 'persist', ParameterDescriptor(
                description='If the models should persist in the GPU when they are inactive. Options: persist, expire'))
        self.declare_parameter(
            'max_audio_recording', 30, ParameterDescriptor(
                description='Max audio recording duration in seconds.'))
        self.declare_parameter(
            'publish_face_expression', True, ParameterDescriptor(
                description='If the node should publish the face expression when listening.')
        )

        self.config_file = self.get_parameter('config_file').value
        self.vad_opt = self.get_parameter('vad').value
        self.whisper_model = self.get_parameter('whisper_model').value
        self.microphone = self.get_parameter('microphone').value
        if not self.microphone:
            self.get_logger().error("Microphone parameter is empty. Please set the microphone parameter to the microphone name.")
            raise ValueError("Microphone parameter is empty. Please set the microphone parameter to the microphone name. ($ros2 run hri_stt list_microphones)")
        if 'persist' == self.get_parameter('gpu_load').value:
            self.unload = False
            try:
                self.model = WhisperModel(self.whisper_model, device="cuda", compute_type="float16")
                self.get_logger().info(f"Whisper '{self.whisper_model}' loaded correctly.")
            except Exception as e:
                self.get_logger().error(f'Fatal Error loading Whisper model during initialization: {str(e)}. Setting expire gpu_load mode.')
                self.unload = True
                self.model = None
        else:
            self.unload = True
            self.model = None
        self.max_audio_recording = self.get_parameter('max_audio_recording').value
        if self.max_audio_recording < 5:
            self.get_logger().warning("Max audio recording time is below 5 seconds. This may affect performance.")
        self.should_pub_face_expression = self.get_parameter('publish_face_expression').value

        # Get the index of the microphone to use
        self.index = get_mic_index(self.microphone)
        while self.index is None:
            # Wait until the microphone is connected
            self.get_logger().error(f"Microphone {self.microphone} not found. Retrying in 5 seconds.")
            time.sleep(5.0)
            self.index = get_mic_index(self.microphone)

        mic_cfg = get_cfg_mic(self.index)
        if mic_cfg is None:
            self.get_logger().error("PyAudio could not obtain microphone configuration.")
            raise ValueError("PyAudio could not obtain microphone configuration.")

        self.channels, self.rate = mic_cfg
        # Silero vad model works with 8000 rate audio, so we need to align the chunk size to be multiple of 8000
        if self.vad_opt == 'silero' and self.rate % 8000 != 0:
            align = max(1, int(round(self.rate / 8000)))
            self.rate = align * 8000
            self.get_logger().info(f"Silero VAD works with 8000 multiples rate audio. Microphone {self.microphone} rate aligned to {self.rate}.")
        self.chunk = get_chunk_mic(self.rate, target_ms=64.0, align=256)
        self.get_logger().info(f'Microphone configuration for {self.microphone} found: {self.channels} channels, {self.rate} rate, {self.chunk} chunk size.')

        # Some microphones (like ReSpeaker 4 Mic Array) have an integrated VAD which can be tuned and configured.
        # In order to tune other microphones adapt the tuning.py file and the following code:
        self.vad = None
        if self.microphone == 'ReSpeaker 4 Mic Array':
            # Get the device
            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev is None:
                raise ValueError('Device not found')
            self.vad = Tuning(dev)
            self.vad.load_config(self.config_file)

        if self.vad_opt == 'silero':
            self.vad, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        elif self.vad_opt == 'mic':
            if not self.vad:
                self.get_logger().error("VAD option 'mic' requires to configure the VAD integrated microphone by editing code."
                                        " Please use the option 'silero' or use a ReSpeaker 4 Mic Array microphone (default).")
                raise ValueError("VAD option 'mic' requires to configure the VAD integrated microphone by editing code."
                                 " Please use the option 'silero' or use a ReSpeaker 4 Mic Array microphone (default).")
        else:
            self.get_logger().error("Invalid VAD option. Options are: silero, mic.")
            raise ValueError("Invalid VAD option. Options are: silero, mic.")

        self.audio_buffer = []
        self.silent_chunks = 0  # Counter for chunks with no voice activity
        self.is_recording = True
        self.audio_lock = threading.Lock()
        self.segment_ready = threading.Event()  # Event to signal that a segment is ready to be processed
        self.stop_event = threading.Event()     # Event to signal stopping the recording due to user cancellation
        self.audio_thread = None

        # Check with fixed number of chunks
        self.chunks_recorded = 0
        self.final_question = ""

        self.max_silent_chunks = int(MAX_SILENCE_DURATION_SECONDS * (self.rate / self.chunk))
        self.max_silent_initial_chunks = int(MAX_INITIAL_SILENCE_DURATION_SECONDS * (self.rate / self.chunk))
        self.silence_chunk_before_sending = int(SILENCE_BEFORE_SENDING * (self.rate / self.chunk))
        self.max_recording_chunks = int(self.max_audio_recording * (self.rate / self.chunk))
        assert (self.silence_chunk_before_sending < self.max_silent_chunks)
        self.get_logger().info(f"Max silent chunks: {self.max_silent_chunks},"
                               f" Silent chunks before sending: {self.silence_chunk_before_sending},"
                               f" Max recording chunks: {self.max_recording_chunks}")

        if self.should_pub_face_expression:
            # QoS Configuration
            qos_1_reliable = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.publisher_face_exp = self.create_publisher(FaceInterface, '/face_expression', qos_1_reliable)
            self.get_logger().info("Publishing face expressions enabled.")

        # Start the action server
        self._action_server = ActionServer(
            self,
            Stt,
            'hri_stt',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        """ Callback to accept or reject the STT action. """
        if goal_request.start is True:
            if not self.audio_thread:
                self.get_logger().info('Received request to start recording.')
                return GoalResponse.ACCEPT
            else:
                self.get_logger().warning('Received request to start recording while another recording is in progress.')
        else:
            self.get_logger().info('Received wrong request to start recording.')
        return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """ Callback to cancel the STT action. """
        self.get_logger().info('Received request to cancel in progress recording.')
        self.is_recording = False
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """ Callback to execute the STT action. """
        # Start multithreaded audio recording
        self.audio_thread = threading.Thread(target=self.audio_thread_func,
                                        args=(goal_handle, self.vad))
        self.audio_thread.start()

        try:
            self.process_audio(goal_handle)
        except Exception as e:
            self.get_logger().error(f"Exception in process audio thread: {e}")
        finally:
            # Wait for the audio thread to finish
            self._on_shutdown()

        result = Stt.Result()
        result.speech = self.final_question
        if rclpy.ok():
            if self.final_question == '':
                self.get_logger().warning('Not able to capture any speech')
                goal_handle.abort()
            else:
                self.get_logger().info(f'Audio recording finished with speech:{result.speech}.')
                goal_handle.succeed()

        # Reset variables to be able to process a new client request
        self.reset_vars()
        return result

    def publish_face_expression(self, expression: str):
        """ Publish a face expression. """
        if rclpy.ok():
            if self.should_pub_face_expression:
                face_msg = FaceInterface()
                if expression == 'listening':
                    face_msg.face = FaceInterface.LISTENING
                else:
                    face_msg.face = FaceInterface.NEUTRAL
                self.publisher_face_exp.publish(face_msg)

    def save_audio_to_wav(self, filename, audio_data, channels, rate):
        """ Save a .wav file with the configuration received."""
        self.get_logger().debug(f'Saving temp file {filename}, with {channels} channels and {rate} rate.')
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(b''.join(audio_data))

    def reset_vars(self):
        """ Reset working variables. """
        self.final_question = ''
        self.chunks_recorded = 0
        self.audio_buffer = []
        self.silent_chunks = 0
        self.is_recording = True
        self.stop_event.clear()
        self.segment_ready.clear()

    def _on_shutdown(self):
        """ Shutdown callback to stop the audio stream and terminate PyAudio. """
        # Signal main thread and audio thread to stop
        self.stop_event.set()
        self.segment_ready.set()

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
            if self.audio_thread.is_alive():
                self.get_logger().warning("Audio thread did not finish in time.")
        self.audio_thread = None

    def _stop_stream(self):
        """
        Stop and close the audio stream and terminate PyAudio from external method.
        Stream and PyAudio are assigned to class variables to be accessible from main thread too.
        """
        try:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                finally:
                    self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None
        try:
            if self.pya is not None:
                self.pya.terminate()
        except Exception:
            pass
        finally:
            self.pya = None

    def process_audio(self, goal_handle):
        """ Process the audio chunks in the buffer and transcribe them. It is executed in a separate thread than the recording. """
        if self.unload:
            try:
                model = WhisperModel(self.whisper_model, device="cuda", compute_type="float16")
            except Exception as e:
                self.get_logger().error(f'Error loading Whisper model: {str(e)}')
                self.is_recording = False
                return
            self.get_logger().info(f"Whisper '{self.whisper_model}' loaded correctly.")
        else:
            model = self.model

        while (not self.stop_event.is_set() and (self.is_recording or len(self.audio_buffer) > 0)):
            # Wait for a speech break to process chunks
            self.get_logger().debug("Waiting for segment_ready in process_thread.")
            self.segment_ready.wait()
            self.get_logger().debug("Clearing segment_ready in process_thread.")
            # Reset the event
            self.segment_ready.clear()

            self.get_logger().debug("Processing buffered audio...")
            audio_data = []

            self.get_logger().debug("Waiting for audio_lock in process_thread.")
            with self.audio_lock:
                self.get_logger().debug("Acquired audio_lock in process_thread.")
                if len(self.audio_buffer) > 0:
                    # Convert the buffer to a single numpy array for processing (Normalizing)
                    audio_data = np.concatenate(self.audio_buffer).astype(np.int16)
                    # Clear the buffer after extracting the data
                    self.audio_buffer = []
                self.get_logger().debug("Free audio_lock in process_thread.")

            if len(audio_data) > 0:
                # Transcribe the audio
                start = time.time()

                # Save the audio to a temporary file to process it and remove it afterwards
                temp_file = f'/tmp/temp_whisper{len(audio_data)}.wav'
                self.save_audio_to_wav(temp_file, audio_data, self.channels, self.rate)
                segments, info = model.transcribe(temp_file, beam_size=5, language="en", condition_on_previous_text=False)
                os.remove(temp_file)

                try:
                    for segment in segments:
                        self.get_logger().info(f"Transcription took {time.time() - start:.3f} seconds. Parcial text: {segment.text}")
                        feedback_msg = Stt.Feedback()
                        feedback_msg.recording = True
                        feedback_msg.partial_speech = segment.text
                        goal_handle.publish_feedback(feedback_msg)
                        self.get_logger().debug(f'Sending fb with partial speech: {feedback_msg.partial_speech}')
                        self.final_question += ' ' + segment.text
                except Exception as e:
                    self.get_logger().error(f"Error processing audio: {e}")
                    # If a CUDA out of memory runtime error occurs, unload the model to force a reload
                    self.unload = True
                    self.model = None
                    self.is_recording = False

            time.sleep(0.1)  # Small delay to avoid busy-waiting
        self.get_logger().debug("Finished processing audio.")

    def _ensure_mono_int16(self, pcm_int16: np.ndarray) -> np.ndarray:
        """Mixes using mean algorithm interleaved PCM int16 into mono."""
        if self.channels > 1:
            pcm_int16 = pcm_int16.reshape(-1, self.channels).mean(axis=1).astype(np.int16)
        return pcm_int16

    def _resample_rate(self, x_f32: np.ndarray) -> np.ndarray:
        """
        Resample float32 audio to target rate if needed. Silero VAD works with 8000 or 16000 rate audio.

        :param x_f32: Input audio in float32 numpy array.
        :return: Resampled audio in float32 numpy array.
        """
        if self.rate in (16000, 8000):
            return x_f32
        elif self.rate > 16000:
            out_rate = 16000
        elif self.rate < 16000:
            out_rate = 8000

        g = gcd(self.rate, 16000)
        up = out_rate // g
        down = self.rate // g
        return resample_poly(x_f32, up=up, down=down).astype(np.float32, copy=False)

    def audio_thread_func(self, goal_handle, vad):
        """ Audio recording thread. Checks for voice activity and records audio chunks. """
        try:
            # Initialize PyAudio
            self.pya = pyaudio.PyAudio()
            self.stream = self.pya.open(channels=self.channels,
                            format=self.pya.get_format_from_width(WIDTH),
                            frames_per_buffer=self.chunk,
                            input_device_index=self.index,
                            input=True,
                            rate=self.rate)
        except:
            self.get_logger().error("Error initializing PyAudio.")
            self.is_recording = False
            return

        self.publish_face_expression('listening')
        self.get_logger().info("*Recording audio*")

        try:
            started = False
            self.is_recording = True
            while not self.stop_event.is_set() and self.is_recording:
                # Read a chunk of data from the stream
                audio_chunk = self.stream.read(self.chunk, exception_on_overflow=False)
                self.chunks_recorded += 1

                # Convert the audio chunk to a numpy array
                audio_i16 = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_i16 = self._ensure_mono_int16(audio_i16)

                # Detect if the chunk contains voice activity
                is_speech = self.chunk_contains_voice(audio_i16, vad)

                if self.chunks_recorded > self.max_recording_chunks:
                    self.get_logger().info("Reached max time conversation.")
                    self.is_recording = False

                self.get_logger().debug("Waiting for audio_lock in audio_thread.")
                with self.audio_lock:
                    self.get_logger().debug("Acquired audio_lock in audio_thread.")
                    if is_speech:
                        if started is False:
                            started = True
                            self.get_logger().info("Detected start of conversation.")
                        # Reset silent chunk counter
                        self.silent_chunks = 0
                        # Append chunk to the buffer
                        self.audio_buffer.append(audio_i16)
                        self.get_logger().debug(f"Appending chunk to buffer {self.chunks_recorded}")
                    else:
                        # Increment silent chunk counter
                        self.silent_chunks += 1
                        if not started:
                            if self.silent_chunks > self.max_silent_initial_chunks:
                                self.is_recording = False
                                self.get_logger().info(f"No voice detected for {MAX_INITIAL_SILENCE_DURATION_SECONDS} seconds. Aborting.")
                        elif started and self.silent_chunks > self.max_silent_chunks:
                            self.get_logger().info(f"Detected end of conversation.")
                            self.is_recording = False
                        else:
                            # Append also silent chunks to the buffer to avoid false negatives and lose audio if voice was detected
                            if len(self.audio_buffer) > 0:
                                self.audio_buffer.append(audio_i16)
                                self.get_logger().debug(f"Appending silent chunk to buffer {self.chunks_recorded}")
                            if self.silent_chunks == self.silence_chunk_before_sending and len(self.audio_buffer) > 0:
                                # If we have buffered audio and detect silence, process the existing buffer
                                self.get_logger().debug("Signaling segment ready for processing...")
                                self.segment_ready.set()
                    self.get_logger().debug("Free audio_lock in audio_thread.")

        except Exception as e:
            self.get_logger().error(f"Exception in audio capture thread: {e}")
            feedback_msg = Stt.Feedback()
            feedback_msg.recording = False
            feedback_msg.partial_speech = ''
            goal_handle.publish_feedback(feedback_msg)

        finally:
            # Stop recording
            self.publish_face_expression('neutral')
            self._stop_stream()
            self.get_logger().debug("Signaling segment_ready in end function.")
            self.segment_ready.set()

    def chunk_contains_voice(self, audio_i16: np.ndarray, vad) -> bool:
        """
        Check if the audio chunk contains voice activity.

        :param audio_chunk: Audio chunk in int16 numpy array. Already in mono.
        :param vad: VAD model or object.
        :return: True if voice activity is detected, False otherwise.
        """
        if self.vad_opt == 'silero':
            audio_f32 = int2float(audio_i16)
            audio_f32 = self._resample_rate(audio_f32)

            # Use exact framing of 512 or 256 samples for Silero VAD
            frame_len = 512 if self.rate >= 16000 else 256
            n_frames = len(audio_f32) // frame_len
            if n_frames == 0:
                # There are not enough samples to process and audio chunk -> no speech
                return False

            # Reshape audio to frames and discard extra samples that do not fit in a full frame of frame_len
            frames = audio_f32[:n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float32)

            self.get_logger().info(f"Silero VAD processing {n_frames} frames of length {frame_len} samples.")
            speech_probs = []
            for i in range(n_frames):
                used_rate = 16000 if self.rate >= 16000 else 8000
                with torch.no_grad():
                    speech_prob = vad(torch.from_numpy(frames[i]), used_rate).item()
                    speech_probs.append(speech_prob)

            self.get_logger().info(f"Silero VAD speech probabilities: {speech_probs}")

            return any(prob >= 0.5 for prob in speech_probs)
        elif self.vad_opt == 'mic':
            if vad.read("SPEECHDETECTED") == 1:
                return True
            else:
                return False
        else:
            self.get_logger().error("Invalid VAD option.")
        return False


def main(args=None):
    rclpy.init(args=args)

    try:
        node = NodeSTT()
    except Exception as e:
        rclpy.logging.get_logger("NodeSTT").error(f"Failed to create NodeSTT: {e}")
        return

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutting down STT node due to external signal...")
        node._on_shutdown()
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
