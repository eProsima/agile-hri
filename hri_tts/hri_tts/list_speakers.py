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

"""
Small utility to parse and list available ALSA playback devices.
Usage examples:
  python list_speakers.py
  ros2 run hri_tts list_speakers
"""

import re
import shutil
import subprocess
import sys


def run_cmd(cmd: list[str]) -> str:
    """Run a command and return stdout text."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        try:
            return out.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return out.decode("latin1", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return ""


def parse_aplay_l(text: str) -> list[dict]:
    """Parse `aplay -l` output into a list of device dicts."""
    devs = []
    pattern = re.compile(
        r"card\s+(\d+):\s*([^\[]+)\[([^\]]+)\],\s*device\s+(\d+):\s*([^\[]+)\[([^\]]*)\]",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        card = int(m.group(1))
        card_short = m.group(2).strip()
        card_long = m.group(3).strip()
        device = int(m.group(4))
        dev_short = m.group(5).strip()
        dev_long = m.group(6).strip()
        devs.append({
            "card": card,
            "device": device,
            "card_short": card_short,
            "card_long": card_long,
            "dev_short": dev_short,
            "dev_long": dev_long,
            "hw": f"hw:{card},{device}",
            "plughw": f"plughw:{card},{device}",
        })
    return devs


def main():
    if shutil.which("aplay") is None:
        print("[Error] `aplay` not found in PATH. Install ALSA utils (`apt install alsa-utils`).", file=sys.stderr)
        sys.exit(1)

    # Get and parse `aplay -l`
    txt_l = run_cmd(["aplay", "-l"])
    if not txt_l:
        print("[Error] No output from `aplay -l`. Do you have playback devices and ALSA set up?", file=sys.stderr)
        sys.exit(2)
    devices = parse_aplay_l(txt_l)

    # Pretty text output
    print("Available playback Hardware Devices:")
    if not devices:
        print("None found.")
    else:
        for d in devices:
            print(f"[card={d['card']},dev={d['device']}] "
                  f"Name: {d['card_long']} - Details: {d['dev_short']} | {d['hw']} | {d['plughw']}")


if __name__ == "__main__":
    main()
