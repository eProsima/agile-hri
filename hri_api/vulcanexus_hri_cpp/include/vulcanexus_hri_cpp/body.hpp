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

#ifndef HRI__BODY_HPP_
#define HRI__BODY_HPP_

#include <memory>
#include <optional>
#include <string>

#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"
#include "hri_msgs/msg/skeleton2_d.hpp"
#include "hri_msgs/msg/skeleton3_d.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"

#include "vulcanexus_hri_cpp/feature_tracker.hpp"
#include "vulcanexus_hri_cpp/types.hpp"

namespace hri {
class Body : public FeatureTracker, public std::enable_shared_from_this<Body>
{
    friend class HRIListener; // for invalidate()

public:

    Body(
            ID id,
            rclcpp::Logger logger,
            rclcpp::CallbackGroup::SharedPtr callback_group,
            const tf2::BufferCore& tf_buffer,
            const std::string& reference_frame);

    virtual ~Body();

    /**
     * @brief If available, returns the normalized 2D region of interest (RoI) of the body.
     *        The coordinates are provided in the original camera's image coordinate space.
     */
    std::optional<cv::Rect2f> roi() const
    {
        return roi_;
    }

    /**
     * @brief Returns the body image, cropped from the source image.
     * @warning Cropped body image is not sent to reduce network congestion.
     */
    std::optional<cv::Mat> cropped() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_,
            "Cropped body image is not sent to reduce network congestion. It can be built using RoI and the original image.");
        return cv::Mat();
    }

    /**
     * @brief Returns the 2D skeleton keypoints.
     *
     * Points coordinates are in the image space of the source image, and
     * normalised between 0.0 and 1.0.
     *
     * The skeleton joints indices follow those defined in
     * http://docs.ros.org/en/api/hri_msgs/html/msg/Skeleton2D.html
     */
    std::optional<SkeletalKeypoints> skeleton() const
    {
        return skeleton_;
    }

    /**
     * @brief Returns the 3D skeleton keypoints.
     *
     * Points coordinates are in the reference frame of the robot.
     *
     * The skeleton joints indices follow those defined in
     * http://docs.ros.org/en/api/hri_msgs/html/msg/Skeleton3D.html
     */
    std::optional<SkeletalKeypoints3D> skeleton3d() const
    {
        return skeleton3d_;
    }

    /**
     * @brief Update the Face with new RoI and landmarks data.
     */
    void update(
            const hri_msgs::msg::NormalizedRegionOfInterest2D& roi_msg,
            const hri_msgs::msg::Skeleton2D& skeleton_msg);

    /**
     * @brief Update the Face with 3D landmarks data.
     */
    void update(
            const hri_msgs::msg::Skeleton3D& skeleton_msg);

    /**
     * @brief Returns the body kinematic description in URDF format.
     *
     * The body kinematic description follows the template defined in
     * http://www.ros.org/reps/rep-0155.html#kinematic-model-of-the-human
     *
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<std::string> bodyDescription() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not use URDF descriptions.");
        return "";
    }

private:
    void invalidate();

    std::optional<cv::Rect2f> roi_;
    std::optional<SkeletalKeypoints> skeleton_;
    std::optional<SkeletalKeypoints3D> skeleton3d_;
};

typedef std::shared_ptr<Body> BodyPtr;
typedef std::shared_ptr<const Body> ConstBodyPtr;

}  // namespace hri

#endif  // HRI__BODY_HPP_
