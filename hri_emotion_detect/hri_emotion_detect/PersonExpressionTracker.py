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

MAX_HISTORY = 10


class PersonExpression:
    """
    Class containing the meta data of a person expression
    """
    def __init__(self):
        self.id = 0
        self.summary = {
            'neutral': 0, 'angry': 0, 'sad': 0, 'happy': 0, 'surprised': 0,
            'disgust': 0, 'scared': 0, 'pleading': 0, 'vulnerable': 0,
            'despaired': 0, 'guilty': 0, 'disappointed': 0, 'embarrassed': 0,
            'fearful': 0, 'skeptical': 0, 'annoyed': 0, 'furious': 0,
            'suspicious': 0, 'rejected': 0, 'bored': 0, 'tired': 0,
            'asleep': 0, 'confused': 0, 'amazed': 0, 'excited': 0
        }
        self.history = []

    def add_expression(self, expression):
        """Add an expression to the summary."""
        if expression in self.summary:
            self.summary[expression] += 1
            # Add expression to history
            if len(self.history) >= MAX_HISTORY:
                self.summary[self.history.pop(0)] -= 1
            self.history.append(expression)
        else:
            print("Unknown expression: ", expression)

    def get_expression(self):
        """Returns the most common expression."""
        return max(self.summary, key=self.summary.get)
