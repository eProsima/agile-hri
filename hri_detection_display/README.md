# HRI Detection Display

The `hri_detection_display` package provides:
- A display to visualize an image with detections from the `hri_face_detect`, `hri_emotion_detect` and `hri_pose_detect`. It has several parameters that allows to filter or tune the detections displayed.
- A window-based image viewer (`node_person_viewer`) that renders the same overlays directly in a dedicated GUI window.
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

## Viewer Node (`node_person_viewer`)

This node mirrors `node_person_display` overlays (IDs, boxes, emotion, skeleton) but shows them in a local OpenCV window.

### Key Parameters

- `image_topic` (default: `/color/image_raw`): Input image topic.
- `window_x`, `window_y`, `window_width`, `window_height`: Initial window geometry.
- `always_on_top` (default: `false`): Best-effort top-most request.
- `bring_to_front` (default: `true`): Best-effort focus request at startup.
- `keep_aspect_ratio` (default: `true`): Preserve image aspect ratio while resizing.
- `no_signal_timeout` (default: `2.0`): Show "No signal" after this many seconds without frames.
- `display_mode`, `allow_half_body`, `allow_back_turned`: Same filtering behavior as `node_person_display`.

### Run Example

```bash
ros2 run hri_detection_display node_person_viewer --ros-args \
  -p image_topic:=/color/image_raw \
  -p window_x:=50 -p window_y:=50 \
  -p window_width:=1280 -p window_height:=720 \
  -p always_on_top:=true -p bring_to_front:=true
```

Close the window or press `q` / `Esc` to stop the node.

Runtime controls:
- `f`: bring window to front (best effort).
- `t`: toggle always-on-top (best effort).
- `w`, `a`, `s`, `d`: move window up/left/down/right by `window_move_step` pixels.

Runtime parameter examples:

```bash
ros2 param set /hri_person_viewer always_on_top true
ros2 param set /hri_person_viewer bring_to_front true
ros2 param set /hri_person_viewer window_x 200
ros2 param set /hri_person_viewer window_y 120
```

### Dependencies

```bash
sudo apt install ros-${ROS_DISTRO}-cv-bridge python3-opencv
```

### Wayland vs Xorg Notes

- On **Xorg**, OpenCV window controls (`moveWindow`, `resizeWindow`, and often top-most/front requests) usually work.
- On **Wayland**, compositors often block strict app-side z-order/focus control for security reasons. `always_on_top` / `bring_to_front` may be ignored.
- Check your session type with:

```bash
echo $XDG_SESSION_TYPE
```

If you need stronger focus/z-order behavior, prefer running an **Xorg session** or use a Qt-based viewer where the compositor allows those hints.
