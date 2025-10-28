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

# FaceInterface.msg
# This message defines the possible visual states (or "modes") for the robot's face interface.
# The robot's facial expression or animation should change according to the mode value received.
# Each mode corresponds to a specific robot activity or idle state.

string SPEAKING = speak      # TTS action
string DANCING = dance       # Dancing action
string LISTENING = listen    # STT action
string NEUTRAL = neutral     # No specific action that requires face interaction

string face neutral          # Current face mode. Should be set to one of the constants above
