# Vulcanexus HRI Pose Detect

A [ROS4HRI](https://wiki.ros.org/hri)-compliant ROS 2 node to perform pose and body detection using
[YOLOv8n-pose](https://docs.ultralytics.com/es/tasks/pose/) from ultralytics.

Although the selected model is able to perform face landmarks detection, only the body landmarks are
represented in order to avoid conflicts with the the
[HRI-Face-Detect](https://github.com/ros4hri/hri_face_detect) package.

## ROS API

### Parameters

All parameters are loaded in the lifecycle `configuration` transition.

- `yolo_model_path` (string, default: "<path_to_share_directory>/models/yolov8n-pose.pt"):
  Path to the YOLOv8 pose detection model

- `processing_rate` (int, default: 30):
  Image processing logic execution best effort rate in Hertz.

- `confidence_threshold` (double, default: 0.75):
  Candidate body detections with confidence lower that this threshold are not
  published.

- `image_scale` (double, default: 1.0):
  The YOLO face detector accepts input image of dynamic size.
  This parameter controls the rescale factor applied to the input image before running the YuNet face detector.
  Lower image scale results in less processing time required and lower detection
  confidences.
  The output data (e.g., RoI) is invariant with this parameter and always refers
  to the original input image size.

- `use_diagnosis` (bool, default: false):
  If true an extra topic will be created to publish dianosis info.

- `diagnostic_period` (float, default: 1.0):
  Period of the dianosis publication if `use_diagnosis` is set to true.

- `id_timeout` (float, default: 7.0):
  Timeout in seconds for the ID manager service. This service is responsible for assigning an ID to the Face. If the service takes longer than this timeout to respond, the ID assignation of the face will be skipped until next iteration.
  Note that this service only takes a few milliseconds to answer in normal conditions.

- `use_depth` (bool, default: false):
  If true, depth images are used to extract depth of each body in the image. This parameters enables the creation
  of depth images subscribers (camera info and camera data) and the corresponding publisher of
  hri_msgs/msg/Skeleton3DList msgs.

- `sync_margin_arg` (float, default: 0.05):
  Time marging for RGB and Depth images synchronization, in seconds. This value is used as a threshold to discard
  messages which timestamps differs more than its value. It is useful to ensure that the information corresponding
  to the depth images corresponds to the one of the RGB images.

- `use_time_offset` (bool, default: false):
  When computing the tracking algorithm to match the results of new inferences with old bodies, a time evaluation is
  performed to ensure that the last body was observed within a time threshold value. If this flag is set to true,
  the first image timestamps received is used as offset to compute the time differences.
  This flag is useful when the image is obtained from a different machine and there might exist clock desynchronization.


### Extra parameters for launch file

- `rgb_camera` (string, default: ""):
  The input RGB camera namespace. Note that this parameter only applies if `rgb_camera_topic` and `rgb_camera_info` are left unset. Otherwise, the value specified in those params will overwrite this namespace.

- `rgb_camera_topic`(string, default: rgb_camera + "/image_raw"):
  The input RGB camera image topic.

- `rgb_camera_info`(string, default: rgb_camera + "/camera_info"):
  The input RGB camera info topic.

- `depth_camera` (string, default: ""):
  The input RGB camera namespace. Note that this parameter only applies if `depth_camera_topic` and `depth_camera_info` are left unset. Otherwise, the value specified in those params will overwrite this namespace.

- `depth_camera_topic`(string, default: rgb_camera + "/image_raw"):
  The input RGB camera image topic.

- `depth_camera_info`(string, default: rgb_camera + "/camera_info"):
  The input RGB camera info topic.

- `log-level` (string, default: info):
  Logging level.

### Topics

This package optimizes the [ROS REP-155](https://ros.org/reps/rep-0155.html) by introducing [keys](https://docs.ros.org/en/rolling/Tutorials/Advanced/Topic-Keys/Topic-Keys-Tutorial.html), compatible with [Fast DDS](https://github.com/eProsima/Fast-DDS).

By using keyed topics, this package is able to provide pose recognition for multiple targets by just publishing in one topic:

- humans/bodies

#### Subscribed

- `color/image_raw` ([sensor_msgs/msg/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg))
- `color/camera_info` ([sensor_msgs/msg/CameraInfo](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg))
- `depth/image_raw` ([sensor_msgs/msg/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg))
- `depth/camera_info` ([sensor_msgs/msg/CameraInfo](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg))

#### Published

- `/humans/bodies` (hri_msgs/msg/Skeleton2DList)
- [Optional] `/humans/faces/skel3D` (hri_msgs/msg/Skeleton3DList)
- [Optional] `/diagnostics` ([diagnostic_msgs/msg/DiagnosticArray](https://github.com/ros2/common_interfaces/blob/jazzy/diagnostic_msgs/msg/DiagnosticArray.msg))

## Execution

```bash
ros2 launch hri_pose_detect pose_detect.launch.py use_depth:=True use_time_offset:=True rgb_camera_topic:=${RGB_CAM_TOPIC} depth_camera_topic:=${DEPTH_CAM_TOPIC} id_timeout:=20.0
```

## Example

For an example of usage check [Vulcanexus HRI Tutorial](https://docs.vulcanexus.org/en/latest/rst/tutorials/hri/pose_detect.html).
