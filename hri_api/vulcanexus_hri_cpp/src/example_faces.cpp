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

#include <functional>
#include <memory>

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "vulcanexus_hri_cpp/vulcanexus_hri.hpp"

using namespace std::chrono_literals;

class ShowFaces : public rclcpp::Node
{
public:

    ShowFaces(std::string topic_name = "/test_image")
        : rclcpp::Node("hri_cpp_example_face"), topic_name_(topic_name) {}

    void init()
    {
        // "shared_from_this()" cannot be used in the constructor!
        hri_listener_ = hri::HRIListener::create(shared_from_this());

        // Subscribe to original image topic to show Data
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            topic_name_, rclcpp::SensorDataQoS(),
            std::bind(&ShowFaces::onImage, this, std::placeholders::_1));

        timer_ = create_wall_timer(10ms, std::bind(&ShowFaces::timer_callback, this));
    }

    void onImage(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try
        {
            last_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        }
        catch (const cv_bridge::Exception& e)
        {
            RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
        }
    }

    void timer_callback()
    {
        if (last_image_.empty())
        {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Waiting for image messages on %s topic...", topic_name_.c_str());
            return;
        }

        cv::Mat vis = last_image_.clone();

        auto faces = hri_listener_->getFaces();
        for (auto const& [face_id, face] : faces)
        {
            if (auto cropped = face->cropped())
            {
                // Cropped image is not sent to avoid network congestion
            }

            if (auto roi = face->roi())
            {
                const auto& rf = *roi;
                cv::Rect ri(cvRound(rf.x * vis.cols), cvRound(rf.y * vis.rows),
                            cvRound(rf.width * vis.cols), cvRound(rf.height * vis.rows));

                cv::rectangle(vis, ri, cv::Scalar(0,255,0), 2);
            }

            if (auto landmarks = face->facialLandmarks())
            {
                for (const auto& [point, value] : landmarks.value())
                {
                    int x = static_cast<int>(value.x * vis.cols);
                    int y = static_cast<int>(value.y * vis.rows);
                    cv::circle(vis, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        cv::imshow("HRI Faces", vis);
        cv::waitKey(1);
    }

private:
    std::string topic_name_;
    std::shared_ptr<hri::HRIListener> hri_listener_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    cv::Mat last_image_;
};

int main(
        int argc,
        char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ShowFaces>();
    node->init();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();

    return 0;
}
