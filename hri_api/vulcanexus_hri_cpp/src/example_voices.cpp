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

#include <functional>
#include <memory>

#include "sensor_msgs/msg/image.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"

#include "hri/hri.hpp"

using namespace std::chrono_literals;

class RequestSpeech : public rclcpp::Node
{
public:

    RequestSpeech()
        : rclcpp::Node("hri_cpp_example_face"){}

    void init()
    {
        // "shared_from_this()" cannot be used in the constructor!
        hri_listener_ = hri::HRIListener::create(shared_from_this());
    }

    std::string request_speech()
    {
        std::string speech;
        try
        {
            RCLCPP_INFO(this->get_logger(), "Requesting speech from user...");
            speech = hri_listener_->getSpeech();
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error requesting speech: %s", e.what());
            throw;
        }
        RCLCPP_INFO(this->get_logger(), "Recognized speech: %s", speech.c_str());
        return speech;
    }

private:
    std::shared_ptr<hri::HRIListener> hri_listener_;
};

int main(
        int argc,
        char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<RequestSpeech>();
    node->init();
    try
    {
        std::string speech = node->request_speech();
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(node->get_logger(), "Error requesting speech: %s", e.what());
    }

    return 0;
}
