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

#include "hri/body.hpp"

#include <functional>
#include <string>

#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"
#include "hri_msgs/msg/skeleton2_d.hpp"

#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"

#include "hri/feature_tracker.hpp"
#include "hri/types.hpp"

namespace hri {

Body::Body(
        ID id,
        rclcpp::Logger logger,
        rclcpp::CallbackGroup::SharedPtr callback_group,
        const tf2::BufferCore& tf_buffer,
        const std::string& reference_frame)
    : FeatureTracker{
            id, "body_", logger, callback_group, tf_buffer, reference_frame}
{
    RCLCPP_DEBUG_STREAM(logger_, "New body detected: " << kId_);
}

Body::~Body()
{
    RCLCPP_DEBUG_STREAM(logger_, "Deleting body " << kId_);
    invalidate();
}

void Body::update(
        const hri_msgs::msg::NormalizedRegionOfInterest2D& roi,
        const hri_msgs::msg::Skeleton2D& skeleton_msg)
{
    roi_.emplace(
        cv::Rect2f{cv::Point2f{roi.xmin, roi.ymin}, cv::Point2f{roi.xmax, roi.ymax}});
    if (!skeleton_)
    {
        skeleton_ = SkeletalKeypoints();
    }
    for (size_t i = 0; i < skeleton_msg.skeleton.size(); ++i)
    {
        const auto& sk = skeleton_msg.skeleton[i];
        // No need to map points as SkeletalKeypoint enum matches Skeleton2D indices
        (*skeleton_)[static_cast<SkeletalKeypoint>(i)] = PointOfInterest{sk.x, sk.y, sk.c};
    }
}

void Body::update(
        const hri_msgs::msg::Skeleton3D& skeleton_msg)
{
    if (!skeleton3d_)
    {
        skeleton3d_ = SkeletalKeypoints3D();
    }
    for (size_t i = 0; i < skeleton_msg.skeleton.size(); ++i)
    {
        const auto& sk = skeleton_msg.skeleton[i];
        // No need to map points as SkeletalKeypoint enum matches Skeleton3D indices
        (*skeleton3d_)[static_cast<SkeletalKeypoint>(i)] = PointOfInterest3D{sk.x, sk.y, sk.z};
    }
}

void Body::invalidate()
{
    roi_.reset();
    skeleton_.reset();
    skeleton3d_.reset();
    FeatureTracker::invalidate();
}

}  // namespace hri
