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

#ifndef HRI__TYPES_HPP_
#define HRI__TYPES_HPP_

#include <array>
#include <string>
#include <map>
#include <type_traits>
#include <variant>

#include "hri_msgs/msg/face2_d.hpp"
#include "hri_msgs/msg/facial_landmarks.hpp"
#include "hri_msgs/msg/normalized_point_of_interest2_d.hpp"
#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"
#include "hri_msgs/msg/skeleton2_d.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace hri {

enum class Expression
{
    kNeutral,
    kAngry,
    kSad,
    kHappy,
    kSurprised,
    kDisgusted,
    kScared,
    kPleading,
    kVulnerable,
    kDespaired,
    kGuilty,
    kDisappointed,
    kEmbarrassed,
    kHorrified,
    kSkeptical,
    kAnnoyed,
    kFurious,
    kSuspicious,
    kRejected,
    kBored,
    kTired,
    kAsleep,
    kConfused,
    kAmazed,
    kExcited,
};

// Not used, but kept for compatibility with ROS4HRI
struct ExpressionVA
{
  float valence;
  float arousal;
};

enum class FacialLandmark
{
    kRightEar = hri_msgs::msg::FacialLandmarks::RIGHT_EAR,
    kRightProfile1 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_1,
    kRightProfile2 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_2,
    kRightProfile3 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_3,
    kRightProfile4 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_4,
    kRightProfile5 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_5,
    kRightProfile6 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_6,
    kRightProfile7 = hri_msgs::msg::FacialLandmarks::RIGHT_PROFILE_7,
    kMenton = hri_msgs::msg::FacialLandmarks::MENTON,
    kLeftEar = hri_msgs::msg::FacialLandmarks::LEFT_EAR,
    kLeftProfile1 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_1,
    kLeftProfile2 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_2,
    kLeftProfile3 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_3,
    kLeftProfile4 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_4,
    kLeftProfile5 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_5,
    kLeftProfile6 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_6,
    kLeftProfile7 = hri_msgs::msg::FacialLandmarks::LEFT_PROFILE_7,
    kRightEyebrowOutside = hri_msgs::msg::FacialLandmarks::RIGHT_EYEBROW_OUTSIDE,
    kRightEyebrow1 = hri_msgs::msg::FacialLandmarks::RIGHT_EYEBROW_1,
    kRightEyebrow2 = hri_msgs::msg::FacialLandmarks::RIGHT_EYEBROW_2,
    kRightEyebrow3 = hri_msgs::msg::FacialLandmarks::RIGHT_EYEBROW_3,
    kRightEyebrowInside = hri_msgs::msg::FacialLandmarks::RIGHT_EYEBROW_INSIDE,
    kRightEyeOutside = hri_msgs::msg::FacialLandmarks::RIGHT_EYE_OUTSIDE,
    kRightEyeTop1 = hri_msgs::msg::FacialLandmarks::RIGHT_EYE_TOP_1,
    kRightEyeTop2 = hri_msgs::msg::FacialLandmarks::RIGHT_EYE_TOP_2,
    kRightEyeInside = hri_msgs::msg::FacialLandmarks::RIGHT_EYE_INSIDE,
    kRightEyeBottom1 = hri_msgs::msg::FacialLandmarks::RIGHT_EYE_BOTTOM_1,
    kRightEyeBottom2 = hri_msgs::msg::FacialLandmarks::RIGHT_EYE_BOTTOM_2,
    kRightPupil = hri_msgs::msg::FacialLandmarks::RIGHT_PUPIL,
    kLeftEyebrowOutside = hri_msgs::msg::FacialLandmarks::LEFT_EYEBROW_OUTSIDE,
    kLeftEyebrow1 = hri_msgs::msg::FacialLandmarks::LEFT_EYEBROW_1,
    kLeftEyebrow2 = hri_msgs::msg::FacialLandmarks::LEFT_EYEBROW_2,
    kLeftEyebrow3 = hri_msgs::msg::FacialLandmarks::LEFT_EYEBROW_3,
    kLeftEyebrowInside = hri_msgs::msg::FacialLandmarks::LEFT_EYEBROW_INSIDE,
    kLeftEyeOutside = hri_msgs::msg::FacialLandmarks::LEFT_EYE_OUTSIDE,
    kLeftEyeTop1 = hri_msgs::msg::FacialLandmarks::LEFT_EYE_TOP_1,
    kLeftEyeTop2 = hri_msgs::msg::FacialLandmarks::LEFT_EYE_TOP_2,
    kLeftEyeInside = hri_msgs::msg::FacialLandmarks::LEFT_EYE_INSIDE,
    kLeftEyeBottom1 = hri_msgs::msg::FacialLandmarks::LEFT_EYE_BOTTOM_1,
    kLeftEyeBottom2 = hri_msgs::msg::FacialLandmarks::LEFT_EYE_BOTTOM_2,
    kLeftPupil = hri_msgs::msg::FacialLandmarks::LEFT_PUPIL,
    kSellion = hri_msgs::msg::FacialLandmarks::SELLION,
    kNose1 = hri_msgs::msg::FacialLandmarks::NOSE_1,
    kNose2 = hri_msgs::msg::FacialLandmarks::NOSE_2,
    kNose = hri_msgs::msg::FacialLandmarks::NOSE,
    kNostril1 = hri_msgs::msg::FacialLandmarks::NOSTRIL_1,
    kNostril2 = hri_msgs::msg::FacialLandmarks::NOSTRIL_2,
    kNostril3 = hri_msgs::msg::FacialLandmarks::NOSTRIL_3,
    kNostril4 = hri_msgs::msg::FacialLandmarks::NOSTRIL_4,
    kNostril5 = hri_msgs::msg::FacialLandmarks::NOSTRIL_5,
    kMouthOuterRight = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_RIGHT,
    kMouthOuterTop1 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_TOP_1,
    kMouthOuterTop2 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_TOP_2,
    kMouthOuterTop3 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_TOP_3,
    kMouthOuterTop4 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_TOP_4,
    kMouthOuterTop5 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_TOP_5,
    kMouthOuterLeft = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_LEFT,
    kMouthOuterBottom1 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_BOTTOM_1,
    kMouthOuterBottom2 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_BOTTOM_2,
    kMouthOuterBottom3 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_BOTTOM_3,
    kMouthOuterBottom4 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_BOTTOM_4,
    kMouthOuterBottom5 = hri_msgs::msg::FacialLandmarks::MOUTH_OUTER_BOTTOM_5,
    kMouthInnerRight = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_RIGHT,
    kMouthInnerTop1 = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_TOP_1,
    kMouthInnerTop2 = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_TOP_2,
    kMouthInnerTop3 = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_TOP_3,
    kMouthInnerLeft = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_LEFT,
    kMouthInnerBottom1 = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_BOTTOM_1,
    kMouthInnerBottom2 = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_BOTTOM_2,
    kMouthInnerBottom3 = hri_msgs::msg::FacialLandmarks::MOUTH_INNER_BOTTOM_3
};

// This array maps Face2D landmark indices to FacialLandmark enum values. It must be updated if the Face2D
// message definition changes.
static constexpr std::array<FacialLandmark, 5> face_to_facial_landmarks = {
    FacialLandmark::kRightEyeInside, // RIGHT_EYE   = 0
    FacialLandmark::kLeftEyeInside,  // LEFT_EYE    = 1
    FacialLandmark::kNose,           // NOSE        = 2
    FacialLandmark::kMouthOuterRight,// RIGHT_MOUTH = 3
    FacialLandmark::kMouthOuterLeft  // LEFT_MOUTH  = 4
};

// Static assert to ensure that the size of face_to_facial_landmarks matches the number of landmarks in Face2D
using LandmarksArrayHRIMsg = decltype(hri_msgs::msg::Face2D{}.landmarks);
static_assert(std::tuple_size_v<LandmarksArrayHRIMsg> == face_to_facial_landmarks.size(),
              "Face2D.landmarks size must match face_to_facial_landmarks");

enum class FeatureType
{
    kInvalid = 0,
    kPerson = (1u << 0), // all known persons, whether or not they are currently seen
    kTrackedPerson = (1u << 1), // only the actively tracked persons
    kFace = (1u << 2),
    kBody = (1u << 3),
    kVoice = (1u << 4)
};  // note that FeatureType values can also be used as bitmasks

inline FeatureType operator &(
        const FeatureType& lhs,
        const FeatureType& rhs)
{
    return static_cast<FeatureType>(
        std::underlying_type<FeatureType>::type(lhs) & std::underlying_type<FeatureType>::type(rhs));
}

inline FeatureType operator |(
        const FeatureType& lhs,
        const FeatureType& rhs)
{
    return static_cast<FeatureType>(
        std::underlying_type<FeatureType>::type(lhs) | std::underlying_type<FeatureType>::type(rhs));
}

typedef std::variant<rclcpp::Node::SharedPtr, rclcpp_lifecycle::LifecycleNode::SharedPtr>
  NodeLikeSharedPtr;

struct PointOfInterest
{
    float x;
    float y;
    float c;
};

struct PointOfInterest3D
{
    double x;
    double y;
    double z;
};

enum class SkeletalKeypoint
{
    kNose = hri_msgs::msg::Skeleton2D::NOSE,
    kNeck = hri_msgs::msg::Skeleton2D::NECK,
    kRightShoulder = hri_msgs::msg::Skeleton2D::RIGHT_SHOULDER,
    kRightElbow = hri_msgs::msg::Skeleton2D::RIGHT_ELBOW,
    kRightWrist = hri_msgs::msg::Skeleton2D::RIGHT_WRIST,
    kLeftShoulder = hri_msgs::msg::Skeleton2D::LEFT_SHOULDER,
    kLeftElbow = hri_msgs::msg::Skeleton2D::LEFT_ELBOW,
    kLeftWrist = hri_msgs::msg::Skeleton2D::LEFT_WRIST,
    kRightHip = hri_msgs::msg::Skeleton2D::RIGHT_HIP,
    kRightKnee = hri_msgs::msg::Skeleton2D::RIGHT_KNEE,
    kRightAnkle = hri_msgs::msg::Skeleton2D::RIGHT_ANKLE,
    kLeftHip = hri_msgs::msg::Skeleton2D::LEFT_HIP,
    kLeftKnee = hri_msgs::msg::Skeleton2D::LEFT_KNEE,
    kLeftAnkle = hri_msgs::msg::Skeleton2D::LEFT_ANKLE,
    kLeftEye = hri_msgs::msg::Skeleton2D::LEFT_EYE,
    kRightEye = hri_msgs::msg::Skeleton2D::RIGHT_EYE,
    kLeftEar = hri_msgs::msg::Skeleton2D::LEFT_EAR,
    kRightEar = hri_msgs::msg::Skeleton2D::RIGHT_EAR
};

typedef std::string ID;
typedef std::map<FacialLandmark, PointOfInterest> FacialLandmarks;
typedef std::map<SkeletalKeypoint, PointOfInterest> SkeletalKeypoints;
typedef std::map<SkeletalKeypoint, PointOfInterest3D> SkeletalKeypoints3D;

}  // namespace hri

#endif  // HRI__TYPES_HPP_
