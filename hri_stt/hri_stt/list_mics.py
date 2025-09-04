# Copyright 2025 Proyectos y Sistemas de Mantenimiento SL (eProsima).
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

import pyaudio

def list_mics() -> None:
    """ List all available microphones in the system. """
    p = pyaudio.PyAudio()

    print("Available microphones:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"- Index {i}: [{device_info['name']}] - Channels: {device_info['maxInputChannels']}, - Rate: {device_info['defaultSampleRate']} Hz")

    p.terminate()


if __name__ == "__main__":
    list_mics()
