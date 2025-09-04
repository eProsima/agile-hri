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

# Audio.msg
# This msg can be used to send audio over ROS 2.
# It can contain either a preset index to point to a local saved audio or real
# audio data as raw bytes with meta-data to allow playing it back.

uint8 preset        # Variable to indicate that this audio refers to a known preset

uint8 num_channels  # Number of channels of the audio sent
int32 sample_rate   # Sample rate of the audio
string subtype      # Format of the audio file (e.g. wav)
uint8 id            # ID to unequivocally identify the audio. It is user responsability
                    # to ensure the uniqueness of this field.
uint8 sentences     # To increase overall efficiency of the audio communication, an Audio.msg can
                    # be used to send a partial audio of a big audio message. The sentences field
                    # indicates the number of partial messages that are needed to complete the whole
                    # audio. A whole audio is normally splitted into sentences.
byte[] data         # Raw data of the Audio message
