# Vulcanexus HRI msgs

This repository contains a set of ROS messages, services, and actions
for Human-Robot Interaction (HRI) applications.

These messages expand the [ROS REP-155](https://ros.org/reps/rep-0155.html) by introducing [keys](https://docs.ros.org/en/rolling/Tutorials/Advanced/Topic-Keys/Topic-Keys-Tutorial.html), compatible with [Fast DDS](https://github.com/eProsima/Fast-DDS).
Keys are ideal for HRI interactions and introduce a huge efficiency improvement to the REP-155.
Keyed topics allow the user to reduce the number of required resources (topics, along with their associated publisher and subscriber) by multiplexing updates of several objects of the same kind into a single resource.

For example, the ROS REP-155 indicates that a ROS 2 Node that detects human faces must create the namespace "/humans/faces/<faceID>/" for each recognized face.
If every face requires to publish 8 differents topics, the detection of a new face will automatically add to the network 8 new publisher and 8 new subscriber, associated to the new topics created.
In an environment with 10 faces, this set up will require 160 pubs/subs to work (_(8 + 8) * num\_topics_).
With keyed topics, the number of publishers and subscribers is fixed, because the same topic is used for every detected face.
This means that no extra entities are created dynamically.
Additionally, in the same environment previously described, only 16 pubs/subs are required, **independently** of the number of faces present.

The same logic applies to other elements of the HRI stack, as bodies or voices.

The following table summarizes the messages, actions and services included in this package.

|                 Name                  | msg | idl | Description |
|---------------------------------------|-----|-----|-------------|
| Audio                                 |  x  |     |Represents audio data.|
| Expression                            |     |  x  |Represents facial expressions.|
| Face2D                                |     |  x  |Represents a 2D face detection.|
| Face2DList                            |  x  |     |Contains a list of 2D face detections.|
| FacialLandmarks                       |     |  x  |Represents facial landmarks.|
| Intent                                |  x  |     |Represents the intent of a person.|
| NormalizedPointOfInterest2D           |  x  |     |Contains the position of a point of interest (typically in an image).|
| NormalizedRegionOfInterest2D          |     |  x  |Contains the top-leftmost and bottom-rightmost coordinates of a region of interest (typically in an image).|
| Pose3D                                |     |  x  |Represents a 3D pose.|
| Skeleton2D                            |     |  x  |Contains a list of 2D skeletal points.|
| Skeleton2DList                        |  x  |     |Contains a list of 2D skeletons.|
| Skeleton3D                            |     |  x  |Contains a list of 3D skeletal points.|
| Skeleton3DList                        |  x  |     |Contains a list of 3D skeletons.|

## Actions

|                 Name                  | Description |
|---------------------------------------|-------------|
| ImageDescription                      |Describes an image.|
| Stt                                   |Speech-to-text action.|
| Tts                                   |Text-to-speech action.|

## Services

|                 Name                  | Description |
|---------------------------------------|-------------|
| PersonID                              |Handles person identification.|
