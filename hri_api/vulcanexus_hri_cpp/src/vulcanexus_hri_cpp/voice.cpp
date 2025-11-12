// Copyright 2025 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vulcanexus_hri_cpp/voice.hpp"

#include <functional>
#include <string>

// #include "hri_msgs/msg/live_speech.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2_ros/buffer.h"

#include "vulcanexus_hri_cpp/feature_tracker.hpp"
#include "vulcanexus_hri_cpp/types.hpp"

namespace hri {

Voice::Voice(
        ID id,
        rclcpp::Logger logger,
        rclcpp::CallbackGroup::SharedPtr callback_group,
        const tf2::BufferCore& tf_buffer,
        const std::string& reference_frame)
    : FeatureTracker{
            id, "voice_", logger, callback_group, tf_buffer, reference_frame}
{
    RCLCPP_DEBUG_STREAM(logger_, "New voice detected: " << kId_);
}

Voice::~Voice()
{
    RCLCPP_DEBUG_STREAM(logger_, "Deleting voice " << kId_);
    invalidate();
}

void Voice::invalidate()
{
    clearCallbacks();
    FeatureTracker::invalidate();
}

}  // namespace hri
