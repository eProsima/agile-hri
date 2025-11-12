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

// Copyright (c) 2015-2019, Carnegie Mellon University. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the {copyright_holder} nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Copied and adapted from https://github.com/dimatura/pyrosmsg

#ifndef VULCANEXUS_HRI_PY__CONVERTERS_HPP_
#define VULCANEXUS_HRI_PY__CONVERTERS_HPP_

#include <set>
#include <string>

#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "opencv2/core.hpp"
#include "pybind11/pybind11.h"

// Declare enum classes not defined in types.hpp for Python bindings
namespace hri
{
enum class EngagementLevel
{
  kDisengaged = 1,
  kEngaging = 2,
  kEngaged = 3,
  kDisengaging = 4
};

enum class FacialActionUnit
{
  kNeutralFace = 0,
  kInnerBrowRaiser = 1,
  kOuterBrowRaiser = 2,
  kBrowLowerer = 3,
  kUpperLidRaiser = 4,
  kCheeckRaiser = 5,
  kLidTightener = 6,
  kLipsTowardEachOther = 7,
  kNoseWrinkler = 8,
  kUpperLipRaiser = 9,
  kNasolabialDeepener = 10,
  kLipCornerPuller = 11,
  kSharpLipPuller = 12,
  kDimpler = 13,
  kLipCornerDepressor = 14,
  kLowerLipDepressor = 15,
  kChinRaiser = 16,
  kLipPucker = 17,
  kTongueShow = 18,
  kLipStretcher = 19,
  kNeckTightener = 20,
  kLipFunneler = 21,
  kLipTightener = 22,
  kLipPressor = 23,
  kLipsPart = 24,
  kJawDrop = 25,
  kMouthStretch = 26,
  kLipSuck = 27,
  kHeadTurnLeft = 28,
  kHeadTurnRight = 29,
  kHeadUp = 30,
  kHeadDown = 31,
  kHeadTiltLeft = 32,
  kHeadTiltRight = 33,
  kHeadForward = 34,
  kHeadBack = 35,
  kEyesTurnLeft = 36,
  kEyesTurnRight = 37,
  kEyesUp = 38,
  kEyesDown = 39,
  kWalleye = 40,
  kCrossEye = 41,
  kEyesPositionedToLookAtOtherPerson = 42,
  kBrownsAndForeheadNotVisible = 43,
  kEyesNotVisible = 44,
  kLowerFaceNotVisible = 45,
  kEntireFaceNotVisible = 46,
  kUnsociable = 47,
  kJawThrust = 48,
  kJawSideways = 49,
  kJawClencher = 50,
  kLipBite = 51,
  kCheekBlow = 52,
  kCheekPuff = 53,
  kCheekSuck = 54,
  kTongueBulge = 55,
  kLipWipe = 56,
  kNostrilDilator = 57,
  kNostrilCompressor = 58,
  kSniff = 59,
  kLidDroop = 60,
  kSlit = 61,
  kEyesClosed = 62,
  kSquint = 63,
  kBlink = 64,
  kWink = 65,
  kSpeech = 66,
  kSwallow = 67,
  kChewing = 68,
  kShoulderShrug = 69,
  kHeadShakeBackAndForth = 70,
  kHeadNodUpAndDown = 71,
  kFlash = 72,
  kPartialFlash = 73,
  kShiverTremble = 74,
  kFastUpDownLook = 75
};

enum class Gender
{
  kFemale = 1,
  kMale = 2,
  kOther = 3
};

struct IntensityConfidence
{
  float intensity;
  float confidence;
};
}  // namespace hri

namespace vulcanexus_hri_py
{

static bool has_all_attribute_fields(pybind11::handle handle, const std::set<std::string> fields)
{
  for (const auto & field : fields) {
    if (!pybind11::hasattr(handle, field.c_str())) {
      return false;
    }
  }
  return true;
}

}  // namespace vulcanexus_hri_py


namespace PYBIND11_NAMESPACE
{
namespace detail
{

template<>
struct type_caster<builtin_interfaces::msg::Time>
{
public:
  PYBIND11_TYPE_CASTER(
    builtin_interfaces::msg::Time, const_name("builtin_interfaces::msg::Time"));

  bool load(handle py_handle, bool)
  {
    if (!vulcanexus_hri_py::has_all_attribute_fields(py_handle, {"sec", "nanosec"})) {
      return false;
    }

    value.sec = (py_handle.attr("sec")).cast<int32_t>();
    value.nanosec = (py_handle.attr("nanosec")).cast<uint32_t>();
    return true;
  }

  static handle cast(builtin_interfaces::msg::Time msg, return_value_policy, handle)
  {
    object py_obj = module::import("builtin_interfaces.msg").attr("Time")();
    py_obj.attr("sec") = pybind11::cast(msg.sec);
    py_obj.attr("nanosec") = pybind11::cast(msg.nanosec);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<std_msgs::msg::Header>
{
public:
  PYBIND11_TYPE_CASTER(std_msgs::msg::Header, const_name("std_msgs::msg::Header"));

  bool load(handle py_handle, bool)
  {
    if (!vulcanexus_hri_py::has_all_attribute_fields(py_handle, {"stamp", "frame_id"})) {
      return false;
    }

    value.stamp = (py_handle.attr("stamp")).cast<builtin_interfaces::msg::Time>();
    value.frame_id = (py_handle.attr("frame_id")).cast<std::string>();
    return true;
  }

  static handle cast(std_msgs::msg::Header msg, return_value_policy, handle)
  {
    object py_obj = module::import("std_msgs.msg").attr("Header")();
    py_obj.attr("stamp") = pybind11::cast(msg.stamp);
    // ROS 2 uses UTF-8 for 'string' in messages and can directly convert to str(), see
    // https://design.ros2.org/articles/wide_strings.html
    // https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html
    py_obj.attr("frame_id") = pybind11::cast(msg.frame_id);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<geometry_msgs::msg::Vector3>
{
public:
  PYBIND11_TYPE_CASTER(geometry_msgs::msg::Vector3, const_name("geometry_msgs::msg::Vector3"));

  bool load(handle py_handle, bool)
  {
    if (!vulcanexus_hri_py::has_all_attribute_fields(py_handle, {"x", "y", "z"})) {
      return false;
    }

    value.x = (py_handle.attr("x")).cast<double>();
    value.y = (py_handle.attr("y")).cast<double>();
    value.z = (py_handle.attr("z")).cast<double>();
    return true;
  }

  static handle cast(geometry_msgs::msg::Vector3 msg, return_value_policy, handle)
  {
    object py_obj = module::import("geometry_msgs.msg").attr("Vector3")();
    py_obj.attr("x") = pybind11::cast(msg.x);
    py_obj.attr("y") = pybind11::cast(msg.y);
    py_obj.attr("z") = pybind11::cast(msg.z);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<geometry_msgs::msg::Quaternion>
{
public:
  PYBIND11_TYPE_CASTER(
    geometry_msgs::msg::Quaternion, const_name("geometry_msgs::msg::Quaternion"));

  bool load(handle py_handle, bool)
  {
    if (!vulcanexus_hri_py::has_all_attribute_fields(py_handle, {"x", "y", "z", "w"})) {
      return false;
    }

    value.x = (py_handle.attr("x")).cast<double>();
    value.y = (py_handle.attr("y")).cast<double>();
    value.z = (py_handle.attr("z")).cast<double>();
    value.w = (py_handle.attr("w")).cast<double>();
    return true;
  }

  static handle cast(geometry_msgs::msg::Quaternion msg, return_value_policy, handle)
  {
    object py_obj = module::import("geometry_msgs.msg").attr("Quaternion")();
    py_obj.attr("x") = pybind11::cast(msg.x);
    py_obj.attr("y") = pybind11::cast(msg.y);
    py_obj.attr("z") = pybind11::cast(msg.z);
    py_obj.attr("w") = pybind11::cast(msg.w);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<geometry_msgs::msg::Transform>
{
public:
  PYBIND11_TYPE_CASTER(geometry_msgs::msg::Transform, const_name("geometry_msgs::msg::Transform"));

  bool load(handle py_handle, bool)
  {
    if (!vulcanexus_hri_py::has_all_attribute_fields(py_handle, {"translation", "rotation"})) {
      return false;
    }

    value.translation = (py_handle.attr("translation")).cast<geometry_msgs::msg::Vector3>();
    value.rotation = (py_handle.attr("rotation")).cast<geometry_msgs::msg::Quaternion>();
    return true;
  }

  static handle cast(geometry_msgs::msg::Transform msg, return_value_policy, handle)
  {
    object py_obj = module::import("geometry_msgs.msg").attr("Transform")();
    py_obj.attr("translation") = pybind11::cast(msg.translation);
    py_obj.attr("rotation") = pybind11::cast(msg.rotation);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<geometry_msgs::msg::TransformStamped>
{
public:
  PYBIND11_TYPE_CASTER(
    geometry_msgs::msg::TransformStamped, const_name("geometry_msgs::msg::TransformStamped"));

  bool load(handle py_handle, bool)
  {
    if (!vulcanexus_hri_py::has_all_attribute_fields(py_handle, {"header", "child_frame_id", "transform"}))
    {
      return false;
    }

    value.header = (py_handle.attr("header")).cast<std_msgs::msg::Header>();
    value.child_frame_id = (py_handle.attr("child_frame_id")).cast<std::string>();
    value.transform = (py_handle.attr("transform")).cast<geometry_msgs::msg::Transform>();
    return true;
  }

  static handle cast(geometry_msgs::msg::TransformStamped msg, return_value_policy, handle)
  {
    object py_obj = module::import("geometry_msgs.msg").attr("TransformStamped")();
    py_obj.attr("header") = pybind11::cast(msg.header);
    // ROS 2 uses UTF-8 for 'string' in messages and can directly convert to str(), see
    // https://design.ros2.org/articles/wide_strings.html
    // https://pybind11.readthedocs.io/en/stable/advanced/cast/strings.html
    py_obj.attr("child_frame_id") = pybind11::cast(msg.child_frame_id);
    py_obj.attr("transform") = pybind11::cast(msg.transform);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<cv::Rect2f>
{
public:
  PYBIND11_TYPE_CASTER(
    cv::Rect2f, const_name("cv::Rect2f"));

  bool load(handle py_handle, bool)
  {
    if (!isinstance<tuple>(py_handle)) {
      return false;
    }
    const auto py_as_tuple = reinterpret_borrow<tuple>(py_handle);
    if (py_as_tuple.size() != 4) {
      return false;
    }
    for (const auto & element : py_as_tuple) {
      if (!isinstance<float>(element)) {
        return false;
      }
    }

    value.x = py_as_tuple[0].cast<float>();
    value.y = py_as_tuple[1].cast<float>();
    value.width = py_as_tuple[2].cast<float>();
    value.height = py_as_tuple[3].cast<float>();
    return true;
  }

  static handle cast(cv::Rect2f rect, return_value_policy, handle)
  {
    auto py_obj = tuple(4);
    py_obj[0] = pybind11::cast(rect.x);
    py_obj[1] = pybind11::cast(rect.y);
    py_obj[2] = pybind11::cast(rect.width);
    py_obj[3] = pybind11::cast(rect.height);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<hri::ExpressionVA>
{
public:
  PYBIND11_TYPE_CASTER(
    hri::ExpressionVA, const_name("hri::ExpressionVA"));

  bool load(handle py_handle, bool)
  {
    if (!isinstance<tuple>(py_handle)) {
      return false;
    }
    const auto py_as_tuple = reinterpret_borrow<tuple>(py_handle);
    if (py_as_tuple.size() != 2) {
      return false;
    }
    for (const auto & element : py_as_tuple) {
      if (!isinstance<float>(element)) {
        return false;
      }
    }

    value.valence = py_as_tuple[0].cast<float>();
    value.arousal = py_as_tuple[1].cast<float>();
    return true;
  }

  static handle cast(hri::ExpressionVA c_obj, return_value_policy, handle)
  {
    auto py_obj = tuple(2);
    py_obj[0] = pybind11::cast(c_obj.valence);
    py_obj[1] = pybind11::cast(c_obj.arousal);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<hri::IntensityConfidence>
{
public:
  PYBIND11_TYPE_CASTER(
    hri::IntensityConfidence, const_name("hri::IntensityConfidence"));

  bool load(handle py_handle, bool)
  {
    if (!isinstance<tuple>(py_handle)) {
      return false;
    }
    const auto py_as_tuple = reinterpret_borrow<tuple>(py_handle);
    if (py_as_tuple.size() != 2) {
      return false;
    }
    for (const auto & element : py_as_tuple) {
      if (!isinstance<float>(element)) {
        return false;
      }
    }

    value.intensity = py_as_tuple[0].cast<float>();
    value.confidence = py_as_tuple[1].cast<float>();
    return true;
  }

  static handle cast(hri::IntensityConfidence c_obj, return_value_policy, handle)
  {
    auto py_obj = tuple(2);
    py_obj[0] = pybind11::cast(c_obj.intensity);
    py_obj[1] = pybind11::cast(c_obj.confidence);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<hri::PointOfInterest>
{
public:
  PYBIND11_TYPE_CASTER(
    hri::PointOfInterest, const_name("hri::PointOfInterest"));

  bool load(handle py_handle, bool)
  {
    if (!isinstance<tuple>(py_handle)) {
      return false;
    }
    const auto py_as_tuple = reinterpret_borrow<tuple>(py_handle);
    if (py_as_tuple.size() != 3) {
      return false;
    }
    for (const auto & element : py_as_tuple) {
      if (!isinstance<float>(element)) {
        return false;
      }
    }

    value.x = py_as_tuple[0].cast<float>();
    value.y = py_as_tuple[1].cast<float>();
    value.c = py_as_tuple[2].cast<float>();
    return true;
  }

  static handle cast(hri::PointOfInterest c_obj, return_value_policy, handle)
  {
    auto py_obj = tuple(3);
    py_obj[0] = pybind11::cast(c_obj.x);
    py_obj[1] = pybind11::cast(c_obj.y);
    py_obj[2] = pybind11::cast(c_obj.c);
    py_obj.inc_ref();
    return py_obj;
  }
};

template<>
struct type_caster<hri::PointOfInterest3D>
{
public:
  PYBIND11_TYPE_CASTER(
    hri::PointOfInterest3D, const_name("hri::PointOfInterest3D"));

  bool load(handle py_handle, bool)
  {
    if (!isinstance<tuple>(py_handle)) {
      return false;
    }
    const auto py_as_tuple = reinterpret_borrow<tuple>(py_handle);
    if (py_as_tuple.size() != 3) {
      return false;
    }
    for (const auto & element : py_as_tuple) {
      if (!isinstance<float>(element)) {
        return false;
      }
    }

    value.x = py_as_tuple[0].cast<float>();
    value.y = py_as_tuple[1].cast<float>();
    value.z = py_as_tuple[2].cast<float>();
    return true;
  }

  static handle cast(hri::PointOfInterest3D c_obj, return_value_policy, handle)
  {
    auto py_obj = tuple(3);
    py_obj[0] = pybind11::cast(c_obj.x);
    py_obj[1] = pybind11::cast(c_obj.y);
    py_obj[2] = pybind11::cast(c_obj.z);
    py_obj.inc_ref();
    return py_obj;
  }
};

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE

#endif  // VULCANEXUS_HRI_PY__CONVERTERS_HPP_
