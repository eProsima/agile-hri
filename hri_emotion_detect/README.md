# Vulcanexus HRI Emotion Detect

The `hri_emotion_detect` package performs emotions detection using the [OpenCV Zoo Facial Expression Recognition model](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition).

It requires to run Vulcanexus HRI Face Detect Node along this node in order to perform emotion recognition.

## ROS API

### Parameters

- `model_expression_detection`(string, default: '<path_to_share_directory>/models/facial_expression_recognition_mobilefacenet_2022july.onnx')
  Path to the facial expression recognition model.

- `backend_id`(string, default: `cv.dnn.DNN_BACKEND_OPENCV`):
  Backend computation ID.

- `target_id`(string, default: `cv.dnn.DNN_TARGET_CPU`):
  Target computation ID.

### Extra parameters for launch file

- `rgb_camera` (string, default: ""):
  The input RGB camera namespace. Note that this parameter only applies if `rgb_camera_topic` and `rgb_camera_info` are left unset. Otherwise, the value specified in those params will overwrite this namespace.

- `rgb_camera_topic`(string, default: rgb_camera + "/image_raw"):
  The input RGB camera image topic.

- `log-level` (string, default: info):
  Logging level.

This package includes an additional launch file that re-uses `hri_face_detect` launch file to run every node needed to perform emotion recognition from one single file.

### Topics

This package optimizes the [ROS REP-155](https://ros.org/reps/rep-0155.html) by introducing [keys](https://docs.ros.org/en/rolling/Tutorials/Advanced/Topic-Keys/Topic-Keys-Tutorial.html), compatible with [Fast DDS](https://github.com/eProsima/Fast-DDS).

By using keyed topics, this package is able to provide emotion recognition for multiple targets by just publishing in one topic:

- /humans/faces/emotion

#### Subscribed

- `color/image_raw` ([sensor_msgs/msg/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg))
- `/humans/faces` (hri_msgs/msg/Face2DList)

#### Published

- `/humans/faces/emotion` (hri_msgs/msg/Expression)

## Execution

```bash
ros2 launch hri_emotion_detect emotion_detect.launch.py rgb_camera_topic:=${RGB_CAM_TOPIC}
```

To launch hri_emotion_detect and hri_face_detect from one single file use:

```bash
ros2 launch hri_emotion_detect emotion_face_detect.launch.py rgb_camera_topic:=${RGB_CAM_TOPIC}
```

## Example

For an example of usage check [Vulcanexus HRI Tutorial](https://docs.vulcanexus.org/en/latest/rst/tutorials/hri/emotion_detect.html).
