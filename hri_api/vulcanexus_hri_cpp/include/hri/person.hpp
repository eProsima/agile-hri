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

#ifndef HRI__PERSON_HPP_
#define HRI__PERSON_HPP_

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/buffer.h"

#include "hri/body.hpp"
#include "hri/face.hpp"
#include "hri/feature_tracker.hpp"
#include "hri/types.hpp"
#include "hri/voice.hpp"

namespace hri {

class HRIListener;
// Forward declarations used in ROS4HRI but not in Vulcanexus HRI
enum class EngagementLevel : int;

/**
 * @brief Class representing a detected person, potentially associated with a face, body, and voice.
 * Vulcanexus HRI automatically assigns the same ID to faces and bodies and does not use `Person` entities.
 * That is, this class is mostly a placeholder for compatibility with ROS4HRI. Vulcanexus HRI
 * do not need to explicitly track or link persons, as faces and bodies are implicitly linked using the same ID.
 */
class Person : public FeatureTracker, public std::enable_shared_from_this<Person>
{
    friend class HRIListener; // for invalidate()

public:

    Person(
            ID id,
            rclcpp::Logger logger,
            rclcpp::CallbackGroup::SharedPtr callback_group,
            std::weak_ptr<const HRIListener> listener,
            const tf2::BufferCore& tf_buffer,
            const std::string& reference_frame);

    virtual ~Person();

    /**
     * @brief Returns a shared pointer to the face of this person, or
     * a nullptr if this person is currently not associated to any detected face.
     */
    ConstFacePtr face() const;

    /**
     * @brief Returns a shared pointer to the body of this person, or
     * a nullptr if this person is currently not associated to any detected body.
     */
    ConstBodyPtr body() const;

    /**
     * @brief Returns a shared pointer to the voice of this person, or
     * a nullptr if this person is currently not associated to any detected voice.
     * @warning Vulcanexus HRI does not publish this data.
     */
    ConstVoicePtr voice() const;

    /**
     * @brief Returns whether the person is anonymous.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<bool> anonymous() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'anonymous'.");
        return std::nullopt;
    }

    /**
     * @brief Returns the engagement status of the person.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<EngagementLevel> engagementStatus() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'engagement status'.");
        return std::nullopt;
    }

    /**
     * @brief Returns the location confidence of the person.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<float> locationConfidence() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'location confidence'.");
        return std::nullopt;
    }

    /**
     * @brief Returns the alias ID of the person.
     * @warning Vulcanexus HRI does not publish this data.
     */
    std::optional<ID> alias() const
    {
        RCLCPP_WARN_STREAM_ONCE(logger_, "Vulcanexus HRI does not publish 'alias'.");
        return std::nullopt;
    }

    std::optional<geometry_msgs::msg::TransformStamped> transform() const override;

private:
    void invalidate();

    std::weak_ptr<const HRIListener> listener_;

    std::optional<ID> face_id_;
    std::optional<ID> body_id_;
};

typedef std::shared_ptr<Person> PersonPtr;
typedef std::shared_ptr<const Person> ConstPersonPtr;

}  // namespace hri

#endif  // HRI__PERSON_HPP_
