# Vulcanexus HRI Speech-to-Text (STT)

A [ROS4HRI](https://wiki.ros.org/hri)-compliant ROS 2 that translates Speech into text.

This package implements a ROS 2 Action Server.
Audio recording is ONLY activated when the action is started after receiving a goal message.
Actions are time-limited, ensuring that no recording longs indefinitely and a result message is always provided.

## Requirements

This package uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with MIT license as main STT model.
It runs locally due to its reduced size and requisites.
Nonetheless, a **Nvidia GPU is required** to make inferences.

Additionally, CUBLAS and CUDNN library paths need to be exported to the environment.
The following command can be used:

```bash
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## ROS API

### Parameters

- `microphone` (string, default: ""):
  Microphone name used by PyAudio to find the microphone device in the system.
  This parameter is mandatory.

- `vad` (string, default: "silero"):
  VAD selection. Wheter to use the integrated VAD of the microphone or load an external one.
  Options: silero, mic.

- `config_file` (string, default: "noise_config.json")
  Microphone configuration file.

- `whisper_model` (string, default: "medium.en"):
  Whisper model selection.

- `gpu_load` (string, default: "persist"):
  If the models should persist in the GPU when the are inactive.
  Options: persist, expire.

- `max_audio_recording` (int, default: 30):
  Max audio recording duration in seconds.

- `publish_face_expression` (bool, default: True):
  If the node should publish the face expression when listening.

### Extra parameters for launch file

- `log-level` (string, default: info):
  Logging level.

### Topics

#### Action Server

- `/hri_stt` (hri_msgs/action/Stt)

#### Published

- [Optional] `/face_expression` (hri_msgs/msg/FaceInterface)

## Execution

```bash
ros2 launch hri_stt stt_server.launch.py microphone:="Your microphone name"
```

## Example

For an example of usage check [Vulcanexus HRI Tutorial](https://docs.vulcanexus.org/en/latest/rst/tutorials/hri/stt/stt.html).

To list all available microphones in the system run:

```bash
ros2 run hri_stt list_microphones
```
