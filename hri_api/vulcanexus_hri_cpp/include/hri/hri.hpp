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

#ifndef HRI__VULCANEXUS_HRI_HPP_
#define HRI__VULCANEXUS_HRI_HPP_

#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include "rclcpp/node_interfaces/node_interfaces.hpp"
#include "hri_msgs/msg/skeleton2_d_list.hpp"
#include "hri_msgs/msg/skeleton3_d_list.hpp"
#include "hri_msgs/msg/face2_d_list.hpp"
#include "hri_msgs/msg/expression.hpp"
#include "hri_msgs/action/stt.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "hri/types.hpp"
#include "hri/face.hpp"
#include "hri/body.hpp"
#include "hri/voice.hpp"
#include "hri/person.hpp"

namespace hri {

using NodeLikeInterfaces =
    rclcpp::node_interfaces::NodeInterfaces<
        rclcpp::node_interfaces::NodeBaseInterface,
        rclcpp::node_interfaces::NodeClockInterface,
        rclcpp::node_interfaces::NodeGraphInterface,
        rclcpp::node_interfaces::NodeLoggingInterface,
        rclcpp::node_interfaces::NodeParametersInterface,
        rclcpp::node_interfaces::NodeTopicsInterface,
        rclcpp::node_interfaces::NodeWaitablesInterface>;
using NodeLikeBaseSharedPtr = rclcpp::node_interfaces::NodeBaseInterface::SharedPtr;
using NodeLikeParamSharedPtr = rclcpp::node_interfaces::NodeParametersInterface::SharedPtr;
using NodeLikeTopicsSharedPtr = rclcpp::node_interfaces::NodeTopicsInterface::SharedPtr;

/**
 * @class HRIListener
 * @brief Class to retrieve HRI features such as faces and bodies
 */
class HRIListener
{
protected:

    // Protected constructor
    explicit HRIListener(
            NodeLikeInterfaces node_like);

    // Delete copy, move constructors and assignment operators
    HRIListener(
            const HRIListener&) = delete;
    HRIListener(
            HRIListener&&) = delete;
    HRIListener& operator =(
            const HRIListener&) = delete;
    HRIListener& operator =(
            HRIListener&&) = delete;

public:

    /**
     * @brief Factory function for the HRIListener class.
     * @param node_like Node interfaces (Node or LifecycleNode). It must be an rvalue reference.
     * @return std::shared_ptr<HRIListener> Shared pointer to the created HRIListener instance
     */
    [[nodiscard]] static std::shared_ptr<HRIListener> create(
            NodeLikeInterfaces node_like)
    {
        return std::shared_ptr<HRIListener>(new HRIListener(node_like));
    }

    /**
     * @brief Factory function for the HRIListener class, accepting a Node or LifecycleNode.
     * @tparam NodeT Type of the node (should derive from rclcpp::Node or rclcpp_lifecycle::LifecycleNode)
     * @param node_sp Shared pointer to the node.
     * @return std::shared_ptr<HRIListener> Shared pointer to the created HRIListener instance
     */
    template<class NodeT>
    [[nodiscard]] static std::shared_ptr<HRIListener> create(std::shared_ptr<NodeT> node_sp)
    {
        NodeLikeInterfaces node_like(*node_sp);
        return std::shared_ptr<HRIListener>(new HRIListener(node_like));
    }

    // Destructor
    ~HRIListener();

    /**
     * @brief Get the currently known faces
     *
     * @return std::map<ID, FacePtr> Map of known faces indexed by their ID
     */
    std::map<ID, FacePtr> getFaces() const;

    /**
     * @brief Register a callback to be called when a new face is detected
     *
     * @param callback The callback function to be called with the FacePtr
     */
    void onFace(
            std::function<void(FacePtr)> callback)
    {
        face_callbacks_.push_back(callback);
    }

    /**
     * @brief Register a callback to be called when a face is lost
     *
     * @param callback The callback function to be called with the face ID
     */
    void onFaceLost(
            std::function<void(ID)> callback)
    {
        face_lost_callbacks_.push_back(callback);
    }

    /**
     * @brief Get the currently known bodies
     *
     * @return std::map<ID, BodyPtr> Map of known bodies indexed by their ID
     */
    std::map<ID, BodyPtr> getBodies() const;

    /**
     * @brief Register a callback to be called when a new body is detected
     *
     * @param callback The callback function to be called with the BodyPtr
     */
    void onBody(
            std::function<void(BodyPtr)> callback)
    {
        body_callbacks_.push_back(callback);
    }

    /**
     * @brief Register a callback to be called when a body is lost
     * This can be used to avoid race conditions between the internal subscribe callbacks and the
     * HRIListener getter functions (e.g., hri::HRIListener::getTrackedPersons(), hri::Face::roi()).
     * If a multithreaded executor is used to freely spin a node which interfaces are used by both
     * HRIListener and by timer/topic/... callbacks which call an HRIListener getter function,
     * then the latter should be added to this callback group.
     *
     * @param callback The callback function to be called with the body ID
     */
    void onBodyLost(
            std::function<void(ID)> callback)
    {
        body_lost_callbacks_.push_back(callback);
    }

    /**
     * @brief Request to the STT action server to start recording the speech.
     * Feedback (partial transcription) is only used for debugging purposes.
     * This method blocks until a speech has been received or timeout.
     * @return Complete speech received
     */
    std::string getSpeech() const;

    /**
     * @brief Set the speech-to-text timeouts
     * @param accept_tmo Timeout for action goal acceptance
     * @param result_tmo Timeout for action result
     */
    void set_stt_timeouts(
            std::chrono::seconds accept_tmo,
            std::chrono::seconds result_tmo)
    {
        accept_timeout_ = accept_tmo;
        result_timeout_ = result_tmo;
    }

    /**
     * @brief Get the currently known voices
     *
     * @return std::map<ID, VoicePtr> Map of known voices indexed by their ID
     * @warning Vulcanexus HRI does not use this data.
     */
    std::map<ID, VoicePtr> getVoices() const;

    /**
     * @brief Register a callback to be called when a new voice is detected
     *
     * @param callback The callback function to be called with the VoicePtr
     * @warning Vulcanexus HRI does not use this data.
     */
    void onVoice(
            std::function<void(VoicePtr)> callback)
    {
        voice_callbacks_.push_back(callback);
    }

    /**
     * @brief Register a callback to be called when a voice is lost
     *
     * @param callback The callback function to be called with the voice ID
     * @warning Vulcanexus HRI does not use this data.
     */
    void onVoiceLost(
            std::function<void(ID)> callback)
    {
        voice_lost_callbacks_.push_back(callback);
    }

    /**
     * @brief Get the currently known persons
     *
     * @return std::map<ID, PersonPtr> Map of known persons indexed by their ID
     * @warning Vulcanexus HRI does not use this data.
     */
    std::map<ID, PersonPtr> getPersons() const;

    /**
     * @brief Register a callback to be called when a new person is detected
     *
     * @param callback The callback function to be called with the PersonPtr
     * @warning Vulcanexus HRI does not use this data.
     */
    void onPerson(
            std::function<void(PersonPtr)> callback)
    {
        person_callbacks_.push_back(callback);
    }

    /**
     * @brief Register a callback to be called when a person is lost
     *
     * @param callback The callback function to be called with the person ID
     * @warning Vulcanexus HRI does not use this data.
     */
    void onPersonLost(
            std::function<void(ID)> callback)
    {
        person_lost_callbacks_.push_back(callback);
    }

    /**
     * @brief Get the currently tracked persons
     *
     * @return std::map<ID, PersonPtr> Map of tracked persons indexed by their ID
     * @warning Vulcanexus HRI does not use this data.
     */
    std::map<ID, PersonPtr> getTrackedPersons() const;

    /**
     * @brief Register a callback to be called when a new tracked person is detected
     *
     * @param callback The callback function to be called with the PersonPtr
     * @warning Vulcanexus HRI does not use this data.
     */
    void onTrackedPerson(
            std::function<void(PersonPtr)> callback)
    {
        person_tracked_callbacks_.push_back(callback);
    }

    /**
     * @brief Register a callback to be called when a tracked person is lost
     *
     * @param callback The callback function to be called with the person ID
     * @warning Vulcanexus HRI does not use this data.
     */
    void onTrackedPersonLost(
            std::function<void(ID)> callback)
    {
        person_tracked_lost_callbacks_.push_back(callback);
    }

    /**
     * @brief Set the reference frame from which the TF transformations of the persons will
     * be returned (via `Person::transform()`).
     *
     * @return std::string The reference frame
     * @warning Vulcanexus HRI does not use this data.
     */
    void setReferenceFrame(
            const std::string& frame)
    {
        reference_frame_ = frame;
    }

    /**
     * @brief Get the mutually exclusive callback group used by HRIListener
     *
     * @return rclcpp::CallbackGroup::SharedPtr The callback group
     */
    rclcpp::CallbackGroup::SharedPtr getCallbackGroup()
    {
        return callback_group_;
    }

    /**
     * @brief Clears internal data, but leaves the registered callbacks
     */
    void clearData()
    {
        faces_.clear();
        bodies_.clear();
        // Following data structures are not used in Vulcanexus HRIListener
        voices_.clear();
        persons_.clear();
        tracked_persons_.clear();
        tf_buffer_.clear();
    }

    /**
     * @brief Clears the registered callbacks
     */
    void clearCallbacks()
    {
        {
            std::lock_guard<std::shared_mutex> lock(face_callbacks_lock_);
            face_callbacks_.clear();
            face_lost_callbacks_.clear();
        }
        {
            std::lock_guard<std::shared_mutex> lock(body_callbacks_lock_);
            body_callbacks_.clear();
            body_lost_callbacks_.clear();
        }
        // Following data structures are not used in Vulcanexus HRIListener
        {
            std::lock_guard<std::shared_mutex> lock(voice_callbacks_lock_);
            voice_callbacks_.clear();
            voice_lost_callbacks_.clear();
        }
        {
            std::lock_guard<std::shared_mutex> lock(persons_callbacks_lock_);
            person_callbacks_.clear();
            person_lost_callbacks_.clear();
        }
        {
            std::lock_guard<std::shared_mutex> lock(persons_tracked_callbacks_lock_);
            person_tracked_callbacks_.clear();
            person_tracked_lost_callbacks_.clear();
        }
    }

private:

    /**
     * @brief Internal callbacks for face messages
     */
    void on_face_message(
            const hri_msgs::msg::Face2DList::ConstSharedPtr msg);
    /**
     * @brief Internal callbacks for skeleton 2D messages
     */
    void on_skeleton_2d_message(
            const hri_msgs::msg::Skeleton2DList::ConstSharedPtr msg);
    /**
     * @brief Internal callbacks for skeleton 3D messages
     */
    void on_skeleton_3d_message(
            const hri_msgs::msg::Skeleton3DList::ConstSharedPtr msg);

    /**
     * @brief Internal callbacks for expression messages
     */
    void on_expression_message(
            const hri_msgs::msg::Expression::ConstSharedPtr msg);

    NodeLikeBaseSharedPtr node_base_interfaces_;
    NodeLikeParamSharedPtr node_param_interfaces_;
    NodeLikeTopicsSharedPtr node_topics_interfaces_;
    rclcpp::Logger logger_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;

    rclcpp::Subscription<hri_msgs::msg::Face2DList>::SharedPtr faces_sub_;
    rclcpp::Subscription<hri_msgs::msg::Skeleton2DList>::SharedPtr skeleton_2d_sub_;
    rclcpp::Subscription<hri_msgs::msg::Skeleton3DList>::SharedPtr skeleton_3d_sub_;
    rclcpp::Subscription<hri_msgs::msg::Expression>::SharedPtr expression_sub_;

    std::map<ID, FacePtr> faces_;
    std::shared_mutex face_callbacks_lock_;
    std::vector<std::function<void(FacePtr)>> face_callbacks_;
    std::vector<std::function<void(ID)>> face_lost_callbacks_;

    std::map<ID, BodyPtr> bodies_;
    std::shared_mutex body_callbacks_lock_;
    std::vector<std::function<void(BodyPtr)>> body_callbacks_;
    std::vector<std::function<void(ID)>> body_lost_callbacks_;

    rclcpp_action::Client<hri_msgs::action::Stt>::SharedPtr stt_client_;
    std::chrono::seconds accept_timeout_{5};
    std::chrono::seconds result_timeout_{40};
    // Following data structures are not used in Vulcanexus HRIListener
    std::map<ID, VoicePtr> voices_;
    std::shared_mutex voice_callbacks_lock_;
    std::vector<std::function<void(VoicePtr)>> voice_callbacks_;
    std::vector<std::function<void(ID)>> voice_lost_callbacks_;

    std::map<ID, PersonPtr> persons_;
    std::shared_mutex persons_callbacks_lock_;
    std::vector<std::function<void(PersonPtr)>> person_callbacks_;
    std::vector<std::function<void(ID)>> person_lost_callbacks_;

    std::map<ID, PersonPtr> tracked_persons_;
    std::shared_mutex persons_tracked_callbacks_lock_;
    std::vector<std::function<void(PersonPtr)>> person_tracked_callbacks_;
    std::vector<std::function<void(ID)>> person_tracked_lost_callbacks_;

    std::string reference_frame_;
    tf2::BufferCore tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace hri

#endif  // HRI__VULCANEXUS_HRI_HPP_
