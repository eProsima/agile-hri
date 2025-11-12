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

#include "vulcanexus_hri_cpp/person.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
// #include "hri_msgs/msg/engagement_level.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2_ros/buffer.h"

#include "vulcanexus_hri_cpp/body.hpp"
#include "vulcanexus_hri_cpp/face.hpp"
#include "vulcanexus_hri_cpp/feature_tracker.hpp"
#include "vulcanexus_hri_cpp/vulcanexus_hri.hpp"
#include "vulcanexus_hri_cpp/types.hpp"
#include "vulcanexus_hri_cpp/voice.hpp"

namespace hri {

Person::Person(
        ID id,
        rclcpp::Logger logger,
        rclcpp::CallbackGroup::SharedPtr callback_group,
        std::weak_ptr<const HRIListener> listener,
        const tf2::BufferCore& tf_buffer,
        const std::string& reference_frame)
    : FeatureTracker{
            id, "person_", logger, callback_group, tf_buffer, reference_frame}
    , listener_(listener)
    , face_id_(id)
    , body_id_(id)
{
    RCLCPP_DEBUG_STREAM(logger_, "New person detected: " << kId_);
}

Person::~Person()
{
    RCLCPP_DEBUG_STREAM(logger_, "Deleting person " << kId_);
    invalidate();
}

ConstFacePtr Person::face() const
{
    auto ret = ConstFacePtr();
    if (auto locked_listener = listener_.lock())
    {
        if (face_id_ && locked_listener->getFaces().count(face_id_.value()))
        {
            ret = locked_listener->getFaces()[*face_id_];
        }
    }
    else
    {
        RCLCPP_WARN_STREAM(
            logger_, "Person " << kId_ << " lost connection to the HRI listener!");
    }
    return ret;
}

ConstBodyPtr Person::body() const
{
    auto ret = ConstBodyPtr();
    if (auto locked_listener = listener_.lock())
    {
        if (body_id_ && locked_listener->getBodies().count(body_id_.value()))
        {
            ret = locked_listener->getBodies()[*body_id_];
        }
    }
    else
    {
        RCLCPP_WARN_STREAM(
            logger_, "Person " << kId_ << " lost connection to the HRI listener!");
    }
    return ret;
}

ConstVoicePtr Person::voice() const
{
    RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI relies on STT and TTS for voice data.");
    return ConstVoicePtr();
}

std::optional<geometry_msgs::msg::TransformStamped> Person::transform() const
{
    return FeatureTracker::transform();
}

void Person::invalidate()
{
    face_id_.reset();
    body_id_.reset();
    FeatureTracker::invalidate();
}

}  // namespace hri
