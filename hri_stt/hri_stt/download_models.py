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
Small utility to pre-download Whisper and VAD models.
Usage examples:
  python download_models.py
  python download_models.py -m small
  python download_models.py -m large-v3 --device cuda
"""

import argparse
import os
import sys
import torch
from faster_whisper import WhisperModel


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download STT model.")
    parser.add_argument(
        "-m", "--model",
        default="medium.en",
        help="Whisper model id (e.g. tiny, base, small, medium, medium.en, large-v3). Default: medium.en"
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device. 'auto' selects cuda if available, else cpu. Default: 'auto'"
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Optionally test the whisper model with a short transcription."
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

    print("[Info] Loading VAD (Silero) from torch.hub ...")
    try:
        # VAD model is downloaded when loaded from torch.hub
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        print("[Info] VAD loaded.")
    except Exception as e:
        print(f"[Warning] Could not load Silero VAD: {e}")

    # Load Whisper model (this triggers download on first use)
    print("[Info] Initializing Whisper model (this may download weights) ...")
    model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8")
    print("[Info] Whisper model ready.")

    # Get the parent path of this file
    file_path = (os.path.dirname(os.path.abspath(__file__)))

    if args.test:
        # Check if model was correctly loaded
        segments, info = model.transcribe(f"{file_path}/../config/0.wav", beam_size=5, language="en", condition_on_previous_text=False)
        for segment in segments:
            print(f"[Info] Correct transcription: {segment.text}")

    print("[Info] Models downloaded.")

if __name__ == '__main__':
    main()
