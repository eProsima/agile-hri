# Vulcanexus HRI Text-to-Speech (TTS)

A [ROS4HRI](https://wiki.ros.org/hri)-compliant ROS 2 that translates text into speech.

This package implements a ROS 2 Action Server (**TTS_pub**) that sends Audio as raw bytes or presets to play
known audio files.

It also includes two other nodes:

- **TTS_sub**: subscribes to the audio topic and plays back the generated audio.
- **TTS_gen**: similar to the main TTS_pub node, but it only saves the generated audio in a WAV file.
  It can be used to create audio files for other purposes. It implements the same action server as the TTS_pub node.

## Requirements

This package uses [Coqui TTS](https://github.com/coqui-ai/TTS) with MPL-2.0 License as TTS engine to make transcriptions.
It runs locally due to its reduced size and requisites.
Nonetheless, a **Nvidia GPU is recommended** to speed up inferences.

## ROS API

### Parameters

#### TTS Publisher (Action Server)

- `model` (string, default: "tts_models/en/ljspeech/tacotron2-DDC"):
  TTS model to use for text-to-speech synthesis.

- `wait_for_finished` (bool, default: True):
  Whether the action should wait for the /tts/finished message before completing (wait for audio playback).
  If this option is set to True, the Action server will wait until a message is received in topic /tts/finished to return
  the result of the Action. It will wait a variable timeout computed from the duration of the audio generated.
  This paramater is designed to facilitate the integration of this node into different pipelines.

#### TTS Subscriber

- `speaker` (string, default: "")
  Speaker name used by aplay to find the sound card device in the system.
  This parameter is mandatory.
  A list of all available speakers can be obtained by running ($ros2 run hri_tts list_speakers)

- `volume` (int, default: 70):
  Volume level (%).

- `publish_face_expression` (bool, default: True):
  If the node should publish the face expression when speaking.

#### TTS Generator

- `file_path` (string, default: "/tmp/audio_generated.wav"):
  Path to the audio file to be saved.

### Extra parameters for launch file

- `log-level` (string, default: info):
  Logging level.

- `launch_pub` (bool, default: True):
  Whether to launch the **TTS_pub** node to publish audio messages.

- `launch_sub` (bool, default: True):
  Whether to launch the **TTS_sub** node to play audio messages.

### Topics

#### Action Server

- `/hri_tts` (hri_msgs/action/Tts) (By TTS_pub & TTS_gen)

#### Published

- `/hri_tts/audio` (hri_msgs/msg/Audio) (By TTS_pub)

- `/hri_tts/finished` (std_msgs/msg/Bool) (By TTS_sub)
- [Optional] `/face_expression` (hri_msgs/msg/FaceInterface) (By TTS_sub)

#### Subscribed

- `/hri_tts/audio` (hri_msgs/msg/Audio) (By TTS_sub)

- `/hri_tts/finished` (std_msgs/msg/Bool) (By TTS_pub)

## Execution

- Publisher:

  ```bash
  ros2 launch hri_tts tts.launch.py wait_for_finished:=False
  ```

- Subscriber:

  ```bash
  ros2 launch hri_tts tts.launch.py launch_pub:=False launch_sub:=True speaker:="Your speaker name"
  ```

- Generator:

  ```bash
  ros2 run hri_tts tts_gen --ros-args -p file_path:=/path/to/generated/file.wav
  ```

## Example

For an example of usage check [Vulcanexus HRI Tutorial](https://docs.vulcanexus.org/en/latest/rst/tutorials/hri/tts/tts.html).

To list all available speakers and playback devices in the system run:

```bash
ros2 run hri_tts list_speakers
```
