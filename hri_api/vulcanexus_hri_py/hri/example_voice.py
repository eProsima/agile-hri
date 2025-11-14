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

from hri import HRIListener

import rclpy


class RequestSpeech():
    def __init__(self):
        # In Python wrapper there is no need to initialize the HRIListener within a ROS 2 Node class
        # The HRIListener creates its own node internally and handles the spinning automatically
        self.hri_listener = HRIListener('vulcanexus_hri_py_example_hri_listener')

    def request_speech(self) -> str:
        """Request speech from the user and return the recognized text."""
        recognized_text = None
        try:
            recognized_text = self.hri_listener.speech()
        except Exception as e:
            print(f'Error while requesting speech: {e}')
        if recognized_text is None:
            print('No speech recognized.')
            return ''
        return recognized_text


def main(args=None):
    rclpy.init(args=args)
    rs = RequestSpeech()
    try:
        speech = rs.request_speech()
        print(f'Recognized speech: {speech}')
    except KeyboardInterrupt:
        print('Shutting down voice example node.')
        rclpy.shutdown()
    except Exception as e:
        print(f'Error in voice example node: {e}')


if __name__ == '__main__':
    main()
