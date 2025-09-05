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

"""
Small utility to pre-download TTS model.
Usage examples:
  python download_models.py
  python download_models.py -m tts_models/en/ljspeech/tacotron2-DDC
"""

import argparse
import torch
import os
from TTS.api import TTS


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download TTS model.")
    parser.add_argument(
        "-m", "--model",
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="TTS model id. Default: tts_models/en/ljspeech/tacotron2-DDC"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. 'auto' selects cuda if available, else cpu. Default: 'auto'"
    )
    return parser.parse_args()


def pick_device(user_choice: str) -> str:
    """Resolve device based on user choice and availability."""
    if user_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return user_choice


def main(args=None):
    args = parse_args()

    device = pick_device(args.device)
    print(f"[Info] Selected model: {args.model}")
    print(f"[Info] Device: {device}")

    print("[Info] Loading TTS model...")
    try:
        tts = TTS(model_name=args.model, progress_bar=True).to(device)
        print("[Info] TTS model loaded.")
    except Exception as e:
        print(f"[Warning] Could not load TTS model: {e}")

    print("[Info] Models downloaded.")

if __name__ == '__main__':
    main()
