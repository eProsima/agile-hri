# Vulcanexus HRI helper library

A wrapper library for the Vulcanexus HRI packages that simplifies accessing data extracted by every package inside the Vulcanexus HRI ecosystem.

This wrapper is fully compatible with the [ROS4HRI helper library](https://github.com/ros4hri/libhri) and exposes an identical API.
The user’s interaction with this helper library remains unchanged.
Differences only affects the internal logic and communication topology, improving efficiency without requiring any changes in how the library is used.

It contains two packages:

- `vulcanexus-hri-cpp`: the C++ implementation package.
- `vulcanexus-hri-py`: the Python implementation package, wrapping the C++ one with TODO.
