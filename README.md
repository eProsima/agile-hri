# Vulcanexus HRI

This repository bundles a complete set of **Vulcanexus Human–Robot Interaction (HRI)** packages, completely compatible and ready-to-use in the Vulcanexus ecosystem (ROS 2).

Each `hri_<package>` directory is an independent package that can be built separately.

For an overview of Vulcanexus HRI concepts and components, see the official [docs](https://docs.vulcanexus.org/en/latest/rst/hri/intro.html).

---

## What is Vulcanexus HRI?

Human-Robot Interaction (HRI) studies how humans and robots can effectively **perceive, communicate, and collaborate**.
It combines computer vision, speech, emotion recognition, and higher-level behavioral models so robots can interact with people naturally and safely.

**Vulcanexus HRI** provides a set of ROS 2–based components for:

- Shared HRI message definitions (hri_msgs)
- Face detection and recognition
- Human pose recognition
- Emotion recognition
- Speech-to-Text (STT)
- Text-to-Speech (TTS)
- Visual displays

All of these components are designed to work together on top of ROS 2 and Fast DDS.

---

## Package overview

### `hri_msgs`

Core **message definitions** used by all Vulcanexus HRI modules.

Main features:

- Defines custom message types for HRI (faces, skeletons, emotions, regions of interest, etc.)
- Uses topic keys (Fast DDS keyed topics) so multiple individuals can be represented efficiently on shared topics
- Provides a common interface layer so all HRI components can interoperate

Any HRI-based application will typically depend on `hri_msgs`.

---

### `hri_face_detect`

**Face detection and recognition** module.

Main features:

- Detects human faces from video streams
- Publishes HRI messages describing 2D face positions and landmarks
- Integrates with `hri_id_manager` to assign consistent IDs to faces across frames and sessions

This is a perception component that lets your robot see and track human faces.

---

### `hri_pose_detect`

**Human body pose and gesture recognition** module.

Main features:

- Detects human skeletons and keypoints from RGB / RGB-D input
- Publishes HRI messages describing 2D (and optionally 3D) body poses
- Integrates with `hri_id_manager` to assign consistent IDs to bodies across frames and sessions

This is a perception component that lets your robot see and track human bodies.

---

### `hri_emotion_detect`

**Emotion recognition** module.

Main features:

- Estimates human emotional state from visual cues
- Publishes emotion-related HRI messages for each tracked person

This enables more **socially aware** behaviors (e.g., adapting interaction or feedback depending on user emotion).

---

### `hri_stt`

**Speech-to-Text (STT)** module.

Main features:

- Converts spoken language (audio input) into text
- Publishes recognized text on ROS 2 topics using HRI-related messages

Use this to let humans **talk to the robot** using natural speech.

---

### `hri_tts`

**Text-to-Speech (TTS)** module.

Main features:

- Converts arbitrary text into speech audio
- Publishes audio on ROS 2 topics or plays it through a speaker
- Can be used from other nodes or via HRI APIs to vocalize responses

Use this to make the robot **speak back** to humans.

---

### `hri_detection_display`

Tools and nodes to **render HRI information on a display**, for example:

- Overlaying face, pose or emotion detections on camera images
- Visualizing tracked humans and their IDs
- Providing a simple on-screen debugging UI for the HRI stack

Useful for demos, debugging, and building robot UIs that show what the robot “perceives”.

---

### `hri_id_manager`

**Identity management** module.

Main features:

- Assigns and manages unique IDs for detected humans
- Ensures the same person keeps the same ID across different HRI modules (face, pose, emotion, etc.)
- Exposes services or topics for mapping detections to identities

This is the glue that lets you **tie together** data from multiple perception modules into coherent “human entities”.

---

### `hri_api`

High-level API and helpers to work with the Vulcanexus HRI stack from your own applications.
It provides a library totally API compatible with the [ROS4HRI](https://github.com/ros4hri) stack.

Typical responsibilities:

- Provide convenience wrappers and utilities around HRI topics/services
- Simplify subscribing to multiple HRI components from a single node
- Offer a more user-friendly interface to HRI features (faces, poses, emotions, STT/TTS, etc.)

---

## Installation & build

A complete guide of installation of Vulcanexus HRI can be found [here](https://docs.vulcanexus.org/en/latest/rst/tutorials/hri/hri_installation.html).
