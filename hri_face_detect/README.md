# Vulcanexus HRI Face Detect

A [ROS4HRI](https://wiki.ros.org/hri)-compliant ROS 2 node to perform fast face detection.

It relies on [YuNet face detector](https://github.com/ShiqiYu/libfacedetection) to perform face detection.
Optionally, a parameter can be configured to use [Mediapipe Face Mesh](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md) to extend the detection obtained by YuNet and get a more precise result.
The former performs well at greater distances (depending on image resolution and image scaling applied) and extracts 5 keypoints.
The latter works only at close distances and extracts all the ROS4HRI-defined landmarks.

## ROS 2 API

### Parameters

All parameters are loaded in the lifecycle `configuration` transition.

- `processing_rate` (int, default: 30):
  Image processing logic execution rate in Hertz.

- `confidence_threshold` (double, default: 0.75):
  Candidate face detections with confidence lower that this threshold are not
  published.

- `image_scale` (double, default: 0.25):
  The YuNet face detector accepts input image of dynamic size.
  This parameter controls the rescale factor applied to the input image before running the YuNet face detector.
  Lower image scale results in less processing time required and lower detection
  confidences.
  The output data (e.g., RoI) is invariant with this parameter and always refers
  to the original input image size.

- `face_mesh` (bool, default: false):
  It enables the additional Mediapipe Face Mesh detection.

- `use_diagnosis` (bool, default: false):
  If true an extra topic will be created to publish diagnostic info.

- `diagnostic_period` (float, default: 1.0):
  Period of the diagnostic publication if `use_diagnosis` is set to true.

- `id_timeout` (float, default: 7.0):
  Timeout in seconds for the ID manager service. This service is responsible for assigning an ID to the Face. If the service takes longer than this timeout to respond, the ID assignation of the face will be skipped until next iteration.
  Note that this service only takes a few milliseconds to answer in normal conditions.

- `use_time_offset` (bool, default: false):
  When computing the tracking algorithm to match the results of new inferences with old bodies, a time evaluation is
  performed to ensure that the last body was observed within a time threshold value. If this flag is set to true,
  the first image timestamp received is used as offset to compute the time differences.
  This flag is useful when the image is obtained from a different machine and there might exist clock desynchronization.

### Extra parameters for launch file

- `rgb_camera` (string, default: "color"):
  The input camera namespace. Note that this parameter only applies if `rgb_camera_topic` and `rgb_camera_info` are left unset. Otherwise, the value specified in those params will overwrite this namespace.

- `rgb_camera_topic` (string, default: "<rgb_camera>/image_raw"):
  The input camera image topic.

- `rgb_camera_info` (string, default: "<rgb_camera>/camera_info"):
  The input camera info topic.

- `log-level` (string, default: info):
  Logging level.

### Topics

This package optimizes the [ROS REP-155](https://ros.org/reps/rep-0155.html) by introducing [keys](https://docs.ros.org/en/rolling/Tutorials/Advanced/Topic-Keys/Topic-Keys-Tutorial.html), compatible with [Fast DDS](https://github.com/eProsima/Fast-DDS).

By using keyed topics, this package is able to provide face recognition for multiple targets by just publishing in one topic:

- /humans/faces

#### Subscribed

- `image_raw` ([sensor_msgs/msg/Image](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/Image.msg))
- `camera_info` ([sensor_msgs/msg/CameraInfo](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/CameraInfo.msg))

#### Published

- `/humans/faces` (hri_msgs/msg/Face2DList)
- [Optional] `/diagnostics` ([diagnostic_msgs/msg/DiagnosticArray](https://github.com/ros2/common_interfaces/blob/jazzy/diagnostic_msgs/msg/DiagnosticArray.msg))
- [Optional] `/humans/faces/mesh` (hri_msgs/msg/FacialLandmarks)

## Execution

```bash
ros2 launch hri_face_detect face_detect.launch.py rgb_camera_topic:=<input camera topic>
```

## Example

For an example of usage check [Vulcanexus HRI Tutorial](https://docs.vulcanexus.org/en/latest/rst/tutorials/hri/face_detect/face_detect.html).
