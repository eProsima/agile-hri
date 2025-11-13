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

#ifndef HRI__FACE_HPP_
#define HRI__FACE_HPP_

#include <memory>
#include <optional>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "hri_msgs/msg/expression.hpp"
#include "hri_msgs/msg/face2_d.hpp"
#include "hri_msgs/msg/facial_landmarks.hpp"
#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"

#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"

#include "hri/feature_tracker.hpp"
#include "hri/types.hpp"

namespace hri {

// Forward declarations used in ROS4HRI but not in Vulcanexus HRI
enum class Gender : int;
enum class FacialActionUnits : int;

class Face : public FeatureTracker, public std::enable_shared_from_this<Face>
{
    friend class HRIListener; // for invalidate()

public:

    Face(
            ID id,
            rclcpp::Logger logger,
            rclcpp::CallbackGroup::SharedPtr callback_group,
            const tf2::BufferCore& tf_buffer,
            const std::string& reference_frame);

    virtual ~Face();

    /**
     * @brief The name of the tf frame that correspond to the gaze direction and
     * orientation of the face.
     */
    std::string gazeFrame() const
    {
        return kGazeFrame_;
    }

    /**
     * @brief Returns the normalized 2D region of interest (RoI) of the face, (if available).
     *
     * The pixel coordinates are provided in the original camera's image coordinate space.
     */
    std::optional<cv::Rect2f> roi() const
    {
        return roi_;
    }

    /**
     * @brief The list of the 70 facial landmarks (2D points, expressed in normalized coordinates),
     * (if available).
     *
     * Constants defined in hri_msgs/msgs/Face2D.idl can be used to access
     * specific points on the face.
     */
    std::optional<FacialLandmarks> facialLandmarks() const
    {
        return landmarks_;
    }

    /**
     * @brief The face expression as a discrete state.
     */
    std::optional<Expression> expression() const
    {
        return expression_;
    }

    /**
     * @brief Returns the face image, if necessary scaled, centered and 0-padded, (if available).
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<cv::Mat> cropped() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_,
                "Cropped face image is not sent to reduce network congestion. It can be built using RoI and the original image.");
        return std::nullopt;
    }

    /**
     * @brief Returns the face image, if necessary scaled, centered and 0-padded, (if available).
     *
     * In addition, the face is rotated so that the eyes are horizontal.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<cv::Mat> aligned() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_,
                "Vulcanexus HRI does not publish aligned face images to reduce network congestion.");
        return std::nullopt;
    }

    /**
     * @brief the list of the facial action units detected on the face.
     *
     * Action units indices follow the Action Unit naming convention by Ekman.
     * List here: https://en.wikipedia.org/wiki/Facial_Action_Coding_System
     *
     * Note that the list is sparse (some indices do not exist in Ekman classification).
     * In addition, depending on the AU detector, some action units might not be
     * detected. In these cases, the confidence value will be 0.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<FacialActionUnits> facialActionUnits() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'facial action units'.");
        return std::nullopt;
    }

    /**
     * @brief Estimated age of this face, if available (eg, the '/softbiometrics' is published).
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<float> age() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'age'.");
        return std::nullopt;
    }

    /**
     * @brief Estimated gender of this face, if available (eg, the '/softbiometrics' is published).
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<Gender> gender() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'gender'.");
        return std::nullopt;
    }

    /**
     * @brief The face expression as a continuous value in the circumplex model space.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<ExpressionVA> expressionVA() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'expressionVA'.");
        return std::nullopt;
    }

    /**
     * @brief The confidence of the face expression estimation.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<float> expressionConfidence() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'expressionConfidence'.");
        return std::nullopt;
    }

    /**
     * @brief Returns the (stamped) 3D transform of the gaze (if available).
     */
    std::optional<geometry_msgs::msg::TransformStamped> gazeTransform() const;

    /**
     * @brief Update the Face with new RoI and landmarks data.
     *
     * @param roi_msg The new region of interest message.
     * @param face_ldmks_msg The new face landmarks message.
     */
    void update(
            const hri_msgs::msg::NormalizedRegionOfInterest2D& roi_msg,
            const hri_msgs::msg::Face2D& face_ldmks_msg);

    /**
     * @brief Update the Face with new expression.
     *
     * @param expression_msg The new expression message.
     */
    void update(
            const hri_msgs::msg::Expression& expression_msg);

private:
    void invalidate();

    std::optional<cv::Rect2f> roi_;
    std::optional<FacialLandmarks> landmarks_;
    std::optional<Expression> expression_;

    const std::string kGazeFrame_;
};

typedef std::shared_ptr<Face> FacePtr;
typedef std::shared_ptr<const Face> ConstFacePtr;

}  // namespace hri

#endif  // HRI__FACE_HPP_
