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

#include "vulcanexus_hri_cpp/vulcanexus_hri.hpp"

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <set>

namespace hri {

HRIListener::HRIListener(
        NodeLikeInterfaces node_like)
    : node_base_interfaces_(node_like.get_node_base_interface())
    , node_param_interfaces_(node_like.get_node_parameters_interface())
    , node_topics_interfaces_(node_like.get_node_topics_interface())
    , logger_(node_like.get_node_logging_interface()->get_logger())
    , reference_frame_("base_link")
{
    RCLCPP_DEBUG_STREAM(logger_, "Initialising the HRI Listener");

    callback_group_ = node_base_interfaces_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = callback_group_;
    auto default_qos = rclcpp::SystemDefaultsQoS();
    rclcpp::SubscriptionOptions tf_options{sub_options};
    tf_options.qos_overriding_options = rclcpp::QosOverridingOptions{
        rclcpp::QosPolicyKind::Depth,
        rclcpp::QosPolicyKind::Durability,
        rclcpp::QosPolicyKind::History,
        rclcpp::QosPolicyKind::Reliability};
    rclcpp::SubscriptionOptions tf_static_options{sub_options};
    tf_static_options.qos_overriding_options = rclcpp::QosOverridingOptions{
        rclcpp::QosPolicyKind::Depth,
        rclcpp::QosPolicyKind::History,
        rclcpp::QosPolicyKind::Reliability};

    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(
        tf_buffer_,
        node_like.get_node_base_interface(),
        node_like.get_node_logging_interface(),
        node_like.get_node_parameters_interface(),
        node_like.get_node_topics_interface(),
        true,
        tf2_ros::DynamicListenerQoS(),
        tf2_ros::StaticListenerQoS(),
        tf_options,
        tf_static_options);

    faces_sub_ = rclcpp::create_subscription<hri_msgs::msg::Face2DList>(
        node_param_interfaces_, node_topics_interfaces_, "/humans/faces", default_qos,
        std::bind(&HRIListener::on_face_message, this, std::placeholders::_1), sub_options);

    skeleton_2d_sub_ = rclcpp::create_subscription<hri_msgs::msg::Skeleton2DList>(
        node_param_interfaces_, node_topics_interfaces_, "/humans/bodies", default_qos,
        std::bind(&HRIListener::on_skeleton_2d_message, this, std::placeholders::_1), sub_options);

    skeleton_3d_sub_ = rclcpp::create_subscription<hri_msgs::msg::Skeleton3DList>(
        node_param_interfaces_, node_topics_interfaces_, "/humans/bodies/skel3D", default_qos,
        std::bind(&HRIListener::on_skeleton_3d_message, this, std::placeholders::_1), sub_options);

    expression_sub_ = rclcpp::create_subscription<hri_msgs::msg::Expression>(
        node_param_interfaces_, node_topics_interfaces_, "/humans/faces/emotion", default_qos,
        std::bind(&HRIListener::on_expression_message, this, std::placeholders::_1), sub_options);

    stt_client_ = rclcpp_action::create_client<hri_msgs::action::Stt>(
        node_like.get_node_base_interface(),
        node_like.get_node_graph_interface(),
        node_like.get_node_logging_interface(),
        node_like.get_node_waitables_interface(),
        "/hri_stt"
    );
}

HRIListener::~HRIListener()
{
    RCLCPP_DEBUG_STREAM(logger_, "Closing the HRI Listener");
    clearCallbacks();
}

std::map<ID, FacePtr> HRIListener::getFaces() const
{
    std::map<ID, FacePtr> result;
    for (auto const& f : faces_)
    {
        result[f.first] = f.second;
    }
    return result;
}

std::map<ID, BodyPtr> HRIListener::getBodies() const
{
    std::map<ID, BodyPtr> result;
    for (auto const& f : bodies_)
    {
        result[f.first] = f.second;
    }
    return result;
}

std::string HRIListener::getSpeech() const
{
    if (!stt_client_)
    {
        throw std::runtime_error("STT client not initialized");
    }

    if (!stt_client_->wait_for_action_server(accept_timeout_))
    {
        throw std::runtime_error("STT server not available within accept timeout");
    }

    auto goal = hri_msgs::action::Stt::Goal();
    goal.start = true;

    rclcpp_action::Client<hri_msgs::action::Stt>::SendGoalOptions opts;
    opts.feedback_callback = [logger = logger_]
                (rclcpp_action::ClientGoalHandle<hri_msgs::action::Stt>::SharedPtr /*gh*/,
                hri_msgs::action::Stt::Feedback::ConstSharedPtr fb)
            {
                if (fb)
                {
                    RCLCPP_DEBUG(logger, "STT feedback: '%s'", fb->partial_speech.c_str());
                }
            };

    // opts.feedback_callback = std::bind(&HRIListener::stt_feedback_cb, this, logger_,
    //         std::placeholders::_1, std::placeholders::_2);

    auto goal_future = stt_client_->async_send_goal(goal, opts);

    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node_base_interfaces_);
    // Wait for goal acceptance
    auto ret = exec.spin_until_future_complete(goal_future, accept_timeout_);
    if (ret == rclcpp::FutureReturnCode::TIMEOUT)
    {
        throw std::runtime_error("Timeout waiting for STT goal acceptance");
    }
    if (ret != rclcpp::FutureReturnCode::SUCCESS)
    {
        throw std::runtime_error("Failed while waiting for STT goal acceptance");
    }

    // Wait for results
    auto goal_handle = goal_future.get();
    if (!goal_handle)
    {
        throw std::runtime_error("STT goal was rejected");
    }
    auto result_future = stt_client_->async_get_result(goal_handle);
    ret = exec.spin_until_future_complete(result_future, result_timeout_);
    if (ret == rclcpp::FutureReturnCode::TIMEOUT)
    {
        (void)stt_client_->async_cancel_goal(goal_handle);
        throw std::runtime_error("Timeout waiting for STT result");
    }
    if (ret != rclcpp::FutureReturnCode::SUCCESS)
    {
        (void)stt_client_->async_cancel_goal(goal_handle);
        throw std::runtime_error("Failed while waiting for STT result");
    }

    auto wrapped = result_future.get();
    switch (wrapped.code)
    {
        case rclcpp_action::ResultCode::SUCCEEDED:
        break;
        case rclcpp_action::ResultCode::ABORTED:
        throw std::runtime_error("STT action aborted");
        case rclcpp_action::ResultCode::CANCELED:
        throw std::runtime_error("STT action canceled");
        default:
        throw std::runtime_error("STT action ended with unknown result code");
    }
    if (!wrapped.result)
    {
        throw std::runtime_error("STT result is null");
    }

    // Final speech
    return wrapped.result->speech;
}


std::map<ID, VoicePtr> HRIListener::getVoices() const
{
    RCLCPP_WARN_STREAM_ONCE(logger_,
        "Vulcanexus HRI relies on `hri_stt` and `hri_tts` for voice recognition and synthesis.");
    return voices_;
}

std::map<ID, PersonPtr> HRIListener::getPersons() const
{
    RCLCPP_WARN_STREAM_ONCE(logger_,
        "Vulcanexus HRI automatically assigns the same ID to faces and bodies and does not use `Person` entities.");
    return persons_;
}

std::map<ID, PersonPtr> HRIListener::getTrackedPersons() const
{
    RCLCPP_WARN_STREAM_ONCE(logger_,
        "Vulcanexus HRI automatically assigns the same ID to faces and bodies and does not use `Person` entities.");
    return tracked_persons_;
}

void HRIListener::on_face_message(
        const hri_msgs::msg::Face2DList::ConstSharedPtr msg)
{
    std::set<ID> received_ids;
    for (const auto& face_msg : msg->landmarks)
    {
        if (face_msg.key.empty())
        {
            // Empty key means no more faces detected in the message
            break;
        }
        received_ids.insert(face_msg.key);
    }

    // Process lost faces first
    for (auto it = faces_.begin(); it != faces_.end(); )
    {
        if (received_ids.find(it->first) == received_ids.end())
        {
            // Face lost
            {
                std::lock_guard<std::shared_mutex> lock(face_callbacks_lock_);
                for (const auto& cb : face_lost_callbacks_)
                {
                    cb(it->first);
                }
            }
            it->second->invalidate();
            it = faces_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Process received faces
    if (msg->landmarks.size() != msg->bboxes.size())
    {
        RCLCPP_ERROR_STREAM(
            logger_,
            "Received face list message with different number of landmarks and bounding boxes: "
                << msg->landmarks.size() << " vs " << msg->bboxes.size());
        return;
    }
    for (size_t i = 0; i < msg->landmarks.size(); ++i)
    {
        const auto& roi_msg = msg->bboxes[i];
        const auto& landmarks = msg->landmarks[i];

        if (landmarks.key.empty())
        {
            // Empty key means no more faces detected in the message
            break;
        }
        auto it = faces_.find(landmarks.key);
        // New face
        if (it == faces_.end())
        {
            FacePtr face;
            face = std::make_shared<Face>(
                landmarks.key, logger_, callback_group_, tf_buffer_, reference_frame_);
            face->update(roi_msg, landmarks);
            faces_[landmarks.key] = face;
            {
                std::lock_guard<std::shared_mutex> lock(face_callbacks_lock_);
                for (const auto& cb : face_callbacks_)
                {
                    cb(face);
                }
            }
        }
        else
        {
            // Update existing face
            it->second->update(roi_msg, landmarks);
        }
    }
}

void HRIListener::on_skeleton_2d_message(
        const hri_msgs::msg::Skeleton2DList::ConstSharedPtr msg)
{
    std::set<ID> received_ids;
    for (const auto& skeleton_msg : msg->skeletons)
    {
        if (skeleton_msg.key.empty())
        {
            // Empty key means no more skeletons detected in the message
            break;
        }
        received_ids.insert(skeleton_msg.key);
    }

    // Process lost bodies first
    for (auto it = bodies_.begin(); it != bodies_.end(); )
    {
        if (received_ids.find(it->first) == received_ids.end())
        {
            // Body lost
            {
                std::lock_guard<std::shared_mutex> lock(body_callbacks_lock_);
                for (const auto& cb : body_lost_callbacks_)
                {
                    cb(it->first);
                }
            }
            it->second->invalidate();
            it = bodies_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Process received bodies
    if (msg->skeletons.size() != msg->bboxes.size())
    {
        RCLCPP_ERROR_STREAM(
            logger_,
            "Received body list message with different number of skeletons and bounding boxes: "
                << msg->skeletons.size() << " vs " << msg->bboxes.size());
        return;
    }
    for (size_t i = 0; i < msg->skeletons.size(); ++i)
    {
        const auto& roi_msg = msg->bboxes[i];
        const auto& skeleton = msg->skeletons[i];

        if (skeleton.key.empty())
        {
            // Empty key means no more skeletons detected in the message
            break;
        }

        auto it = bodies_.find(skeleton.key);
        // New body
        if (it == bodies_.end())
        {
            BodyPtr body;
            body = std::make_shared<Body>(
                skeleton.key, logger_, callback_group_, tf_buffer_, reference_frame_);
            body->update(roi_msg, skeleton);
            bodies_[skeleton.key] = body;
            {
                std::lock_guard<std::shared_mutex> lock(body_callbacks_lock_);
                for (const auto& cb : body_callbacks_)
                {
                    cb(body);
                }
            }
        }
        else
        {
            // Update existing body
            it->second->update(roi_msg, skeleton);
        }
    }
}

void HRIListener::on_skeleton_3d_message(
        const hri_msgs::msg::Skeleton3DList::ConstSharedPtr msg)
{
    // Lost and new callbacks are handled in the 2D skeleton callback
    // This method only updates the 3D skeleton data of existing bodies

    // Process received 3D bodies
    for (size_t i = 0; i < msg->skeletons.size(); ++i)
    {
        const auto& skeleton = msg->skeletons[i];

        if (skeleton.key.empty())
        {
            // Empty key means no more skeletons detected in the message
            break;
        }

        auto it = bodies_.find(skeleton.key);
        // If 2D body has not been detected yet, skip it and wait for the 2D callback
        if (it != bodies_.end())
        {
            // Update existing body
            it->second->update(skeleton);
        }
    }
}

void HRIListener::on_expression_message(
        const hri_msgs::msg::Expression::ConstSharedPtr msg)
{
    // This method only updates the Expression data of existing faces
    auto it = faces_.find(msg->key);
    if (it != faces_.end())
    {
        // Update existing face
        it->second->update(*msg);
    }
}

}  // namespace hri
