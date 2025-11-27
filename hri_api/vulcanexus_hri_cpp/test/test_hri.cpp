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

#include <thread>
#include <chrono>
#include <memory>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "hri_msgs/msg/expression.hpp"
#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"
#include "hri_msgs/msg/face2_d_list.hpp"
#include "hri_msgs/msg/face2_d.hpp"
#include "hri_msgs/msg/skeleton2_d_list.hpp"
#include "hri_msgs/msg/skeleton2_d.hpp"
#include "hri_msgs/msg/audio.hpp"
#include "hri_msgs/action/stt.hpp"

#include "gtest/gtest.h"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

#include "hri/hri.hpp"
#include "hri/face.hpp"
#include "hri/person.hpp"
#include "hri/voice.hpp"

using namespace std::chrono_literals;

class HRITest : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        rclcpp::init(0, NULL);
    }

    static void TearDownTestSuite()
    {
        rclcpp::shutdown();
    }

    void SetUp() override
    {
        tester_node_ = rclcpp::Node::make_shared("tester_node_");
        hri_executor_ = rclcpp::executors::MultiThreadedExecutor::make_shared();
        hri_node_ = rclcpp::Node::make_shared("hri_node");
        hri_executor_->add_node(hri_node_);
        hri_listener_ = hri::HRIListener::create(hri_node_);
    }

    void TearDown() override
    {
        hri_listener_.reset();
        hri_node_.reset();
        hri_executor_.reset();
        tester_node_.reset();
    }

    void spin(std::chrono::nanoseconds hri_timeout = 100ms)
    {
        hri_executor_->spin_all(hri_timeout);
    }

    const rclcpp::QoS kQoSLatched_{rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable()};
    rclcpp::Executor::SharedPtr hri_executor_;
    rclcpp::Node::SharedPtr hri_node_;
    rclcpp::Node::SharedPtr tester_node_;
    std::shared_ptr<hri::HRIListener> hri_listener_;
};

TEST_F(HRITest, GetFaces)
{
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto empty_msg = hri_msgs::msg::Face2DList();
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";
    auto face_b = hri_msgs::msg::Face2D();
    face_b.key = "B";

    ASSERT_EQ(faces_pub->get_subscription_count(), 1U);
    EXPECT_EQ(hri_listener_->getFaces().size(), 0U);

    face_list_msg.landmarks[0] = face_a;
    faces_pub->publish(face_list_msg);
    spin();
    auto faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 1U);
    ASSERT_TRUE(faces.count("A"));
    EXPECT_EQ(faces["A"]->id(), "A");

    faces_pub->publish(face_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getFaces().size(), 1U);

    face_list_msg.landmarks[1] = face_b;
    faces_pub->publish(face_list_msg);
    spin();
    faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 2U);
    EXPECT_TRUE(faces.count("A"));
    EXPECT_TRUE(faces.count("B"));

    face_list_msg.landmarks = empty_msg.landmarks;
    face_list_msg.landmarks[0] = face_b;
    faces_pub->publish(face_list_msg);
    spin();
    faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 1U);
    EXPECT_FALSE(faces.count("A"));
    ASSERT_TRUE(faces.count("B"));

    face_list_msg.landmarks = empty_msg.landmarks;
    faces_pub->publish(face_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getFaces().size(), 0U);
    // check face B is not used anymore by hri_listener_!
    EXPECT_EQ(faces["B"].use_count(), 1U);

    hri_listener_.reset();
    EXPECT_EQ(faces_pub->get_subscription_count(), 0U);
    EXPECT_FALSE(faces["B"]->valid());
}

TEST_F(HRITest, GetFacesRoi)
{
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto empty_msg = hri_msgs::msg::Face2DList();
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";
    auto face_b = hri_msgs::msg::Face2D();
    face_b.key = "B";
    auto roi_msg = hri_msgs::msg::NormalizedRegionOfInterest2D();
    roi_msg.xmin = 0.1;
    roi_msg.ymin = 0;
    roi_msg.xmax = 1;
    roi_msg.ymax = 1;

    // Publish only a Face2DList with Face A
    face_list_msg.landmarks[0] = face_a;
    face_list_msg.bboxes[0] = roi_msg;
    faces_pub->publish(face_list_msg);
    spin();
    auto faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 1U);
    ASSERT_TRUE(faces.count("A"));
    EXPECT_EQ(faces["A"]->id(), "A");
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().x, 0.1f);
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().y, 0.0f);
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().width, 0.9f);
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().height, 1.0f);

    // Publish only a Face2DList with Face B. Face A should be removed.
    face_list_msg.landmarks[0] = face_b;
    roi_msg.xmax = 0.4;
    roi_msg.ymax = 0.2;
    face_list_msg.bboxes[0] = roi_msg;
    faces_pub->publish(face_list_msg);
    spin();
    faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 1U);
    ASSERT_FALSE(faces.count("A"))
        << "Face A should have been removed.";
    ASSERT_TRUE(faces.count("B"))
        << "Face B should have been added.";
    EXPECT_EQ(faces["B"]->id(), "B");
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().x, 0.1f);
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().y, 0.0f);
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().width, 0.3f);
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().height, 0.2f);

    // Publish a Face2DList with both Face A and B, updating B's RoI and adding A's new RoI
    roi_msg.xmax = 0.5;
    face_list_msg.bboxes[0] = roi_msg;  // Face B updated RoI
    roi_msg.xmin = 0.0;
    roi_msg.ymin = 0.0;
    roi_msg.xmax = 0.6;
    roi_msg.ymax = 0.7;
    face_list_msg.landmarks[1] = face_a;  // Face A added again
    face_list_msg.bboxes[1] = roi_msg;
    faces_pub->publish(face_list_msg);
    spin();

    faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 2U);
    ASSERT_FALSE(faces["A"] == nullptr);
    ASSERT_FALSE(faces["B"] == nullptr);
    ASSERT_TRUE(faces["A"]->roi());
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().x, 0.0f);
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().y, 0.0f);
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().width, 0.6f);
    EXPECT_FLOAT_EQ(faces["A"]->roi().value().height, 0.7f);
    ASSERT_TRUE(faces["B"]->roi());
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().x, 0.1f);
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().y, 0.0f);
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().width, 0.4f);
    EXPECT_FLOAT_EQ(faces["B"]->roi().value().height, 0.2f);
}

TEST_F(HRITest, GetBodies)
{
    auto bodies_pub = tester_node_->create_publisher<hri_msgs::msg::Skeleton2DList>(
        "/humans/bodies", 1);
    auto empty_msg = hri_msgs::msg::Skeleton2DList();
    auto body_list_msg = hri_msgs::msg::Skeleton2DList();
    auto body_a = hri_msgs::msg::Skeleton2D();
    body_a.key = "A";
    auto body_b = hri_msgs::msg::Skeleton2D();
    body_b.key = "B";

    ASSERT_EQ(bodies_pub->get_subscription_count(), 1U);
    EXPECT_EQ(hri_listener_->getBodies().size(), 0U);

    body_list_msg.skeletons[0] = body_a;
    bodies_pub->publish(body_list_msg);
    spin();
    auto bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 1U);
    ASSERT_TRUE(bodies.count("A"));
    EXPECT_EQ(bodies["A"]->id(), "A");

    bodies_pub->publish(body_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getBodies().size(), 1U);

    body_list_msg.skeletons[1] = body_b;
    bodies_pub->publish(body_list_msg);
    spin();
    bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 2U);
    EXPECT_TRUE(bodies.count("A"));
    EXPECT_TRUE(bodies.count("B"));

    body_list_msg.skeletons = empty_msg.skeletons;
    body_list_msg.skeletons[0] = body_b;
    bodies_pub->publish(body_list_msg);
    spin();
    bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 1U);
    EXPECT_FALSE(bodies.count("A"));
    ASSERT_TRUE(bodies.count("B"));

    body_list_msg.skeletons = empty_msg.skeletons;
    bodies_pub->publish(body_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getBodies().size(), 0U);
    // check face B is not used anymore by hri_listener_!
    EXPECT_EQ(bodies["B"].use_count(), 1U);

    hri_listener_.reset();
    EXPECT_EQ(bodies_pub->get_subscription_count(), 0U);
    EXPECT_FALSE(bodies["B"]->valid());
}

TEST_F(HRITest, GetBodies3D)
{
    auto bodies_pub = tester_node_->create_publisher<hri_msgs::msg::Skeleton2DList>(
        "/humans/bodies", 1);
    auto bodies3d_pub = tester_node_->create_publisher<hri_msgs::msg::Skeleton3DList>(
        "/humans/bodies/skel3D", 1);
    auto empty_msg = hri_msgs::msg::Skeleton2DList();
    auto body_list_msg = hri_msgs::msg::Skeleton2DList();
    auto body3d_list_msg = hri_msgs::msg::Skeleton3DList();
    auto body3d_a = hri_msgs::msg::Skeleton3D();
    auto body_a = hri_msgs::msg::Skeleton2D();
    body3d_a.key = "A";
    body_a.key = "A";
    auto body3d_b = hri_msgs::msg::Skeleton3D();
    auto body_b = hri_msgs::msg::Skeleton2D();
    body3d_b.key = "B";
    body_b.key = "B";

    ASSERT_EQ(bodies3d_pub->get_subscription_count(), 1U);
    EXPECT_EQ(hri_listener_->getBodies().size(), 0U);

    // Publish Skeleton 2D message and then Skeleton 3D message for body A
    body3d_list_msg.skeletons[0] = body3d_a;
    body_list_msg.skeletons[0] = body_a;
    bodies_pub->publish(body_list_msg);
    spin();
    bodies3d_pub->publish(body3d_list_msg);
    spin();
    auto bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 1U);
    ASSERT_TRUE(bodies.count("A"));
    EXPECT_EQ(bodies["A"]->id(), "A");

    // Publish again 3D skeleton for body A -> No change expected
    bodies3d_pub->publish(body3d_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getBodies().size(), 1U);

    // Publish 3D skeleton for body B -> Body B ignored as no 2D skeleton exists yet
    body3d_list_msg.skeletons[1] = body3d_b;
    bodies3d_pub->publish(body3d_list_msg);
    spin();
    bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 1U);
    EXPECT_TRUE(bodies.count("A"));
    EXPECT_FALSE(bodies.count("B"));

    // Publish 2D and 3D skeleton for body B -> Body B should be added now
    body_list_msg.skeletons[1] = body_b;
    bodies_pub->publish(body_list_msg);
    spin();
    bodies3d_pub->publish(body3d_list_msg);
    spin();
    bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 2U);
    EXPECT_TRUE(bodies.count("A"));
    EXPECT_TRUE(bodies.count("B"));

    // Remove body A by publishing only body B in 2D skeletons. Receiving body B 3D again should be ignored
    body_list_msg.skeletons = empty_msg.skeletons;
    body_list_msg.skeletons[0] = body_b;
    bodies_pub->publish(body_list_msg);
    spin();
    bodies3d_pub->publish(body3d_list_msg);
    spin();
    bodies = hri_listener_->getBodies();
    EXPECT_EQ(bodies.size(), 1U);
    EXPECT_FALSE(bodies.count("A"));
    ASSERT_TRUE(bodies.count("B"));

    body_list_msg.skeletons = empty_msg.skeletons;
    bodies_pub->publish(body_list_msg);
    spin();
    bodies3d_pub->publish(body3d_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getBodies().size(), 0U);
    // check face B is not used anymore by hri_listener_!
    EXPECT_EQ(bodies["B"].use_count(), 1U);

    hri_listener_.reset();
    EXPECT_EQ(bodies3d_pub->get_subscription_count(), 0U);
    EXPECT_FALSE(bodies["B"]->valid());
}

TEST_F(HRITest, GetVoices)
{
    auto voices_pub = tester_node_->create_publisher<hri_msgs::msg::Audio>(
        "/humans/voices/tracked", 1);

    ASSERT_EQ(voices_pub->get_subscription_count(), 0U);

    // Assert that a warning is logged after calling getVoices()
    testing::internal::CaptureStderr();
    EXPECT_EQ(hri_listener_->getVoices().size(), 0U);
    auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Vulcanexus HRI relies on `hri_stt` and `hri_tts` for voice recognition and synthesis."),
              std::string::npos);
}

TEST_F(HRITest, FaceCallbacks)
{
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto empty_msg = hri_msgs::msg::Face2DList();
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";
    auto face_b = hri_msgs::msg::Face2D();
    face_b.key = "B";

    ASSERT_EQ(faces_pub->get_subscription_count(), 1U);
    EXPECT_EQ(hri_listener_->getFaces().size(), 0U);

    uint8_t cb_triggered = 0;
    hri_listener_->onFace(
        [&](hri::FacePtr face) {
            cb_triggered++;
        });
    hri_listener_->onFaceLost(
        [&](hri::ID key) {
            cb_triggered--;
        });

    ASSERT_EQ(cb_triggered, 0);
    face_list_msg.landmarks[0] = face_a;
    faces_pub->publish(face_list_msg);
    spin();
    auto faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 1U);
    ASSERT_TRUE(faces.count("A"));
    EXPECT_EQ(faces["A"]->id(), "A");
    EXPECT_EQ(cb_triggered, 1);

    face_list_msg.landmarks[1] = face_b;
    faces_pub->publish(face_list_msg);
    spin();
    faces = hri_listener_->getFaces();
    EXPECT_EQ(faces.size(), 2U);
    EXPECT_TRUE(faces.count("A"));
    EXPECT_TRUE(faces.count("B"));
    EXPECT_EQ(cb_triggered, 2);

    ASSERT_TRUE(cb_triggered);
    face_list_msg.landmarks = empty_msg.landmarks;
    faces_pub->publish(face_list_msg);
    spin();
    EXPECT_EQ(hri_listener_->getFaces().size(), 0U);
    EXPECT_EQ(cb_triggered, 0);
}

TEST_F(HRITest, GetKnownPersons)
{
    auto persons_pub = tester_node_->create_publisher<hri_msgs::msg::Face2D>(
        "/humans/persons/known", 1);
    auto ids_msg = hri_msgs::msg::Face2D();

    ASSERT_EQ(persons_pub->get_subscription_count(), 0U);

    // Assert that a warning is logged after calling getPersons()
    testing::internal::CaptureStderr();
    EXPECT_EQ(hri_listener_->getPersons().size(), 0U);
    auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Vulcanexus HRI automatically assigns the same ID to faces and bodies and does not use `Person` entities."),
              std::string::npos);
}

TEST_F(HRITest, GetTrackedPersons)
{
    auto tracked_persons_pub = tester_node_->create_publisher<hri_msgs::msg::Face2D>(
        "/humans/persons/tracked", 1);
    auto ids_msg = hri_msgs::msg::Face2D();

    ASSERT_EQ(tracked_persons_pub->get_subscription_count(), 0U);

    // Assert that a warning is logged after calling getPersons()
    testing::internal::CaptureStderr();
    EXPECT_EQ(hri_listener_->getTrackedPersons().size(), 0U);
    auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Vulcanexus HRI automatically assigns the same ID to faces and bodies and does not use `Person` entities."),
              std::string::npos);
}

TEST_F(HRITest, GetFacesExpression)
{
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto expression_pub = tester_node_->create_publisher<hri_msgs::msg::Expression>(
        "/humans/faces/emotion", 1);
    auto empty_msg = hri_msgs::msg::Face2DList();
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";
    auto face_b = hri_msgs::msg::Face2D();
    face_b.key = "B";
    auto expression_msg = hri_msgs::msg::Expression();

    // Publish both face IDs
    face_list_msg.landmarks[0] = face_a;
    face_list_msg.landmarks[1] = face_b;
    faces_pub->publish(face_list_msg);
    spin();
    ASSERT_EQ(expression_pub->get_subscription_count(), 1U);
    auto faces = hri_listener_->getFaces();

    // Test reception of an expression
    expression_msg.key = "A";
    expression_msg.expression = hri_msgs::msg::Expression::HAPPY;
    expression_pub->publish(expression_msg);
    spin();
    ASSERT_TRUE(faces["A"]->expression().has_value());
    EXPECT_EQ(faces["A"]->expression().value(), hri::Expression::kHappy);

    // Assert that a warning after calling expressionConfidence()
    testing::internal::CaptureStderr();
    ASSERT_FALSE(faces["A"]->expressionConfidence().has_value());
    auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Vulcanexus HRI does not publish 'expressionConfidence'."), std::string::npos);

    // Test reception of an expression change
    expression_msg.expression = hri_msgs::msg::Expression::SAD;
    expression_pub->publish(expression_msg);
    spin();
    ASSERT_TRUE(faces["A"]->expression().has_value());
    EXPECT_EQ(faces["A"]->expression().value(), hri::Expression::kSad);

    expression_msg.key = "B";
    expression_msg.expression = hri_msgs::msg::Expression::ANGRY;
    expression_pub->publish(expression_msg);
    spin();
    ASSERT_TRUE(faces["B"]->expression().has_value());
    EXPECT_EQ(faces["B"]->expression().value(), hri::Expression::kAngry);
    EXPECT_EQ(faces["A"]->expression().value(), hri::Expression::kSad);
}

TEST_F(HRITest, Image)
{
    // Vulcanexus HRI does not publish face cropped images, so we test that the callback
    // issues a warning when trying to access them.
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";

    // Publish face ID
    face_list_msg.landmarks[0] = face_a;
    faces_pub->publish(face_list_msg);
    spin();
    auto faces = hri_listener_->getFaces();
    ASSERT_EQ(faces_pub->get_subscription_count(), 1U);
    ASSERT_TRUE(faces.count("A"));

    // Assert that a warning is logged after calling face->cropped()
    testing::internal::CaptureStderr();
    auto cropped = faces["A"]->cropped();
    auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Cropped face image is not sent to reduce network congestion. It can be built using RoI and the original image."),
              std::string::npos);
    EXPECT_FALSE(cropped.has_value());

    // Assert that a warning is logged after calling face->aligned()
    testing::internal::CaptureStderr();
    auto aligned = faces["A"]->aligned();
    output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Vulcanexus HRI does not publish aligned face images to reduce network congestion."),
              std::string::npos);
    EXPECT_FALSE(aligned.has_value());
}

TEST_F(HRITest, FacialActionUnits)
{
    // Vulcanexus HRI does not publish face cropped images, so we test that the callback
    // issues a warning when trying to access them.
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";

    // Publish face ID
    face_list_msg.landmarks[0] = face_a;
    faces_pub->publish(face_list_msg);
    spin();
    auto faces = hri_listener_->getFaces();
    ASSERT_EQ(faces_pub->get_subscription_count(), 1U);
    ASSERT_TRUE(faces.count("A"));

    // Assert that a warning is logged after calling face->facialActionUnits()
    testing::internal::CaptureStderr();
    auto fau = faces["A"]->facialActionUnits();
    auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("Vulcanexus HRI does not publish 'facial action units'."), std::string::npos);
    EXPECT_FALSE(fau.has_value());
}

TEST_F(HRITest, FacialLandmarks)
{
    // Vulcanexus HRI does not publish face cropped images, so we test that the callback
    // issues a warning when trying to access them.
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "A";
    auto norm_poi_nose = hri_msgs::msg::NormalizedPointOfInterest2D();
    norm_poi_nose.x = 0.4;
    norm_poi_nose.y = 0.2;
    face_a.landmarks[hri_msgs::msg::Face2D::NOSE] = norm_poi_nose;
    auto norm_poi_le = hri_msgs::msg::NormalizedPointOfInterest2D();
    norm_poi_le.x = 0.6;
    norm_poi_le.y = 0.8;
    face_a.landmarks[hri_msgs::msg::Face2D::LEFT_EYE] = norm_poi_le;
    auto point = hri::PointOfInterest();

    // Publish face ID
    face_list_msg.landmarks[0] = face_a;
    faces_pub->publish(face_list_msg);
    spin();
    auto faces = hri_listener_->getFaces();
    ASSERT_EQ(faces_pub->get_subscription_count(), 1U);
    ASSERT_TRUE(faces.count("A"));

    ASSERT_TRUE(faces["A"]->facialLandmarks());
    point = (*faces["A"]->facialLandmarks())[hri::FacialLandmark::kNose];
    EXPECT_FLOAT_EQ(point.x, norm_poi_nose.x);
    EXPECT_FLOAT_EQ(point.y, norm_poi_nose.y);
    point = (*faces["A"]->facialLandmarks())[hri::FacialLandmark::kLeftEyeInside];
    EXPECT_FLOAT_EQ(point.x, norm_poi_le.x);
    EXPECT_FLOAT_EQ(point.y, norm_poi_le.y);
}

TEST_F(HRITest, SkeletalKeypoints)
{
    auto bodies_pub = tester_node_->create_publisher<hri_msgs::msg::Skeleton2DList>(
        "/humans/bodies", 1);
    auto body_list_msg = hri_msgs::msg::Skeleton2DList();
    auto body_a = hri_msgs::msg::Skeleton2D();
    body_a.key = "A";

    // Define a couple of 2D keypoints (similar to FacialLandmarks test)
    auto poi_nose = hri_msgs::msg::NormalizedPointOfInterest2D();
    poi_nose.x = 0.3f;
    poi_nose.y = 0.5f;
    poi_nose.c = 0.8f;
    body_a.skeleton[hri_msgs::msg::Skeleton2D::NOSE] = poi_nose;

    auto poi_le = hri_msgs::msg::NormalizedPointOfInterest2D();
    poi_le.x = 0.6f;
    poi_le.y = 0.8f;
    poi_le.c = 0.9f;
    body_a.skeleton[hri_msgs::msg::Skeleton2D::LEFT_EYE] = poi_le;

    // Publish body A with keypoints
    body_list_msg.skeletons[0] = body_a;
    bodies_pub->publish(body_list_msg);
    spin();

    auto bodies = hri_listener_->getBodies();
    ASSERT_EQ(bodies_pub->get_subscription_count(), 1U);
    ASSERT_TRUE(bodies.count("A"));

    auto point = hri::PointOfInterest();
    ASSERT_TRUE(bodies["A"]->skeleton());

    point = (*bodies["A"]->skeleton())[hri::SkeletalKeypoint::kNose];
    EXPECT_FLOAT_EQ(point.x, poi_nose.x);
    EXPECT_FLOAT_EQ(point.y, poi_nose.y);
    EXPECT_FLOAT_EQ(point.c, poi_nose.c);

    point = (*bodies["A"]->skeleton())[hri::SkeletalKeypoint::kLeftEye];
    EXPECT_FLOAT_EQ(point.x, poi_le.x);
    EXPECT_FLOAT_EQ(point.y, poi_le.y);
    EXPECT_FLOAT_EQ(point.c, poi_le.c);
}

TEST_F(HRITest, SkeletalKeypoints3D)
{
    // Create body via 2D list, then publish 3D keypoints and validate
    auto bodies2d_pub = tester_node_->create_publisher<hri_msgs::msg::Skeleton2DList>(
        "/humans/bodies", 1);
    auto bodies3d_pub = tester_node_->create_publisher<hri_msgs::msg::Skeleton3DList>(
        "/humans/bodies/skel3D", 1);

    auto body2d_list_msg = hri_msgs::msg::Skeleton2DList();
    auto body3d_list_msg = hri_msgs::msg::Skeleton3DList();
    auto body2d_a = hri_msgs::msg::Skeleton2D();
    auto body3d_a = hri_msgs::msg::Skeleton3D();
    body2d_a.key = "A";
    body3d_a.key = "A";

    // Define a couple of 3D keypoints
    body3d_a.skeleton[hri_msgs::msg::Skeleton3D::NOSE].x = 0.1;
    body3d_a.skeleton[hri_msgs::msg::Skeleton3D::NOSE].y = 0.2;
    body3d_a.skeleton[hri_msgs::msg::Skeleton3D::NOSE].z = 1.0;

    body3d_a.skeleton[hri_msgs::msg::Skeleton3D::LEFT_EYE].x = -0.3;
    body3d_a.skeleton[hri_msgs::msg::Skeleton3D::LEFT_EYE].y = 0.4;
    body3d_a.skeleton[hri_msgs::msg::Skeleton3D::LEFT_EYE].z = 0.9;

    // First publish 2D list to create body A
    body2d_list_msg.skeletons[0] = body2d_a;
    bodies2d_pub->publish(body2d_list_msg);
    spin();

    // Then publish 3D skeleton to update the body with 3D keypoints
    body3d_list_msg.skeletons[0] = body3d_a;
    bodies3d_pub->publish(body3d_list_msg);
    spin();

    auto bodies = hri_listener_->getBodies();
    ASSERT_TRUE(bodies.count("A"));
    ASSERT_TRUE(bodies["A"]->skeleton3d());

    auto p3 = hri::PointOfInterest3D();

    p3 = (*bodies["A"]->skeleton3d())[hri::SkeletalKeypoint::kNose];
    EXPECT_DOUBLE_EQ(p3.x, 0.1);
    EXPECT_DOUBLE_EQ(p3.y, 0.2);
    EXPECT_DOUBLE_EQ(p3.z, 1.0);

    p3 = (*bodies["A"]->skeleton3d())[hri::SkeletalKeypoint::kLeftEye];
    EXPECT_DOUBLE_EQ(p3.x, -0.3);
    EXPECT_DOUBLE_EQ(p3.y, 0.4);
    EXPECT_DOUBLE_EQ(p3.z, 0.9);
}

TEST_F(HRITest, GazeTransform)
{
    // Publish a face first (using Face2DList like other tests) then create TFs and test gaze transform reference frame change
    auto faces_pub = tester_node_->create_publisher<hri_msgs::msg::Face2DList>(
        "/humans/faces", 1);
    auto face_list_msg = hri_msgs::msg::Face2DList();
    auto face_a = hri_msgs::msg::Face2D();
    face_a.key = "f1";
    face_list_msg.landmarks[0] = face_a;

    // Static TF broadcaster for frames
    auto static_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(tester_node_);
    auto transform_msg = geometry_msgs::msg::TransformStamped();

    // Set initial reference frame transform world->base_link
    hri_listener_->setReferenceFrame("base_link");
    transform_msg.header.stamp = tester_node_->now();
    transform_msg.header.frame_id = "world";
    transform_msg.child_frame_id = "base_link";
    transform_msg.transform.translation.x = -1.0;
    transform_msg.transform.translation.y = 0.0;
    transform_msg.transform.translation.z = 0.0;
    transform_msg.transform.rotation.w = 1.0;
    static_broadcaster->sendTransform(transform_msg);
    spin();

    // Publish face list to create face f1
    faces_pub->publish(face_list_msg);
    spin();
    auto f1 = hri_listener_->getFaces()["f1"];
    EXPECT_FALSE(f1->gazeTransform()) << "no gaze transform should be available yet";

    // Publish gaze frame transform base_link->gaze_f1
    transform_msg.header.stamp = tester_node_->now();
    transform_msg.header.frame_id = "base_link";
    transform_msg.child_frame_id = "gaze_f1";
    transform_msg.transform.translation.x = 2.0; // relative to base_link
    transform_msg.transform.translation.y = 0.0;
    transform_msg.transform.translation.z = 0.0;
    transform_msg.transform.rotation.w = 1.0;
    static_broadcaster->sendTransform(transform_msg);
    spin();

    ASSERT_TRUE(f1->gazeTransform()) << "the gaze transform should be available";
    auto t = f1->gazeTransform().value();
    EXPECT_EQ(t.child_frame_id, "gaze_f1");
    EXPECT_EQ(t.header.frame_id, "base_link");
    EXPECT_FLOAT_EQ(t.transform.translation.x, 2.0f);

    // Change reference frame to gaze_f1 and verify transform is identity
    hri_listener_->setReferenceFrame("gaze_f1");
    ASSERT_TRUE(f1->gazeTransform());
    t = f1->gazeTransform().value();
    EXPECT_EQ(t.child_frame_id, "gaze_f1");
    EXPECT_EQ(t.header.frame_id, "gaze_f1");
    EXPECT_FLOAT_EQ(t.transform.translation.x, 0.f);
}


int main(int argc, char ** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
