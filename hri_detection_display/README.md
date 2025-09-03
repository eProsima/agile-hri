# HRI Detection Display

The `hri_detection_display` package provides:
- A display to visualize an image with detections from the `hri_face_detect`, `hri_emotion_detect` and `hri_pose_detect`. It has several parameters that allows to filter or tune the detections displayed.
- A 3D human display to be used with the Orbbecc Astra camera.
- Camera launch files with several configuration options.

## Parameters

- `rgb_camera`(default: color): The input rgb camera namespace.
- `rgb_camera_topic` (default: `<rgb_camera>`/image_raw): The input rgb camera image topic.
- `log-level` (default: info): Logging level.
- `processing_rate` (default: 30): Best effort frequency for processing input images.
- `display_mode` (default: all): Display mode to be used. Available options: [all, both, body, face].
    * `all`: Display all detections.
    * `both`: Display only persons which have a matching body and face.
    * `body`: Display only bodies.
    * `face`: Display only faces.
- `allow_half_body` (default: True): Allow displaying bodies that are not entirely visible. \
                      A body is considered whole if at least the head and one shoulder, hip and knee are visible.
- `allow_back_turned` (default: True): Allow displaying bodies that are not facing the camera.
- `rviz_config_file`(default: [rviz/view.rviz](rviz/view.rviz)): Path to the RViz config file to use.

## Subscribed Topics

- `color/image_raw` (*sensor_msgs/Image*): Subscribes to the image stream.
- `/humans/bodies` (*hri_msgs/Skeleton2DList*): Subscribes to the detected bodies.
- `/humans/faces` (*hri_msgs/Face2DList*): Subscribes to the detected faces.
- `/humans/faces/emotion` (*hri_msgs/Expression*): Subscribes to the detected emotions for faces.

## Published Topics

- `/humans/detection` (*sensor_msgs/Image*): Publishes the annotated image with detections required.

## Usage

To launch the node with RViz configured to show the images, run:

```bash
ros2 launch hri_detection_display person_detection_display.launch.py
```

To run the `hri_detection_display` node by itself, use the following command:

```bash
ros2 run hri_detection_display node_person_display
```
