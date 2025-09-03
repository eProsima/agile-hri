# HRI ID Mangager

The `hri_id_manager` package provides a node that handles the ID assignation for the detections of the `hri_face_detect` and `hri_pose_detect` packages.
The execution of this node is *MANDATORY* in order to be able to run the pose and face detection nodes of their respective packages.
It is responsible of assigning IDs to each face and body detected.
It registers all detections of each node and checks if new faces or bodies belong to the same person.
A match between a `face` and a `body` is considered when:
- A maximum distance threshold is not surpassed.
- A minimum % area overlapping is produced between the face and the body ROIs.
- A recent update of the entity has been registered.

## Parameters

- `log-level` (default: info): Logging level.
- `deterministic_ids` (default: False): Enable use of non-random increasing body IDs.

## Subscribed Topics

- `/humans/bodies` (*hri_msgs/msgs/Skeleton2DList*): Subscribes to the detected bodies.
- `/humans/faces` (*hri_msgs/msgs/Face2DList*): Subscribes to the detected faces.

## Services

- `/assign_id` (*hri_msgs/srv/PersonID*): Service that provides an ID for the entity (face/body) requested.

## Usage

To launch the node run:

```bash
ros2 launch hri_id_manager id_manager.launch.py
```
