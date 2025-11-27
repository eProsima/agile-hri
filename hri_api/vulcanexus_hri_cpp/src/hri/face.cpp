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

#include "hri/face.hpp"

#include <functional>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "hri_msgs/msg/expression.hpp"
#include "hri_msgs/msg/facial_landmarks.hpp"
#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"
#include "hri_msgs/msg/normalized_point_of_interest2_d.hpp"

#include "magic_enum.hpp"
#include "rclcpp/rclcpp.hpp"

#include "hri/feature_tracker.hpp"
#include "hri/types.hpp"

namespace hri {

Face::Face(
        ID id,
        rclcpp::Logger logger,
        rclcpp::CallbackGroup::SharedPtr callback_group,
        const tf2::BufferCore& tf_buffer,
        const std::string& reference_frame)
    : FeatureTracker{
            id, "face_", logger, callback_group, tf_buffer, reference_frame}
    , kGazeFrame_("gaze_" + kId_)
{
    RCLCPP_DEBUG_STREAM(logger_, "New face detected: " << kId_);
}

Face::~Face()
{
    RCLCPP_DEBUG_STREAM(logger_, "Deleting face " << kId_);
    invalidate();
}

std::optional<geometry_msgs::msg::TransformStamped> Face::gazeTransform() const
{
    return transformFromReference(gazeFrame());
}

void Face::update(
        const hri_msgs::msg::NormalizedRegionOfInterest2D& roi,
        const hri_msgs::msg::Face2D& face_ldmks)
{
    roi_.emplace(
        cv::Rect2f{cv::Point2f{roi.xmin, roi.ymin}, cv::Point2f{roi.xmax, roi.ymax}});

    if (!landmarks_)
    {
        landmarks_ = FacialLandmarks();
    }
    for (size_t i = 0; i < face_ldmks.landmarks.size(); ++i)
    {
        const auto& lm = face_ldmks.landmarks[i];
        const FacialLandmark fl = face_to_facial_landmarks[i];
        (*landmarks_)[fl] = PointOfInterest{lm.x, lm.y, lm.c};
    }
}

void Face::update(
            const hri_msgs::msg::Expression& expression_msg)
{
    auto expression = magic_enum::enum_cast<Expression>(
        "k" + expression_msg.expression, magic_enum::case_insensitive);
    if (!expression)
    {
        RCLCPP_WARN_STREAM(logger_, "Received invalid expression: " << expression_msg.expression);
    }
    expression_ = expression;
}

void Face::invalidate()
{
    roi_.reset();
    landmarks_.reset();
    expression_.reset();
    FeatureTracker::invalidate();
}

}  // namespace hri
