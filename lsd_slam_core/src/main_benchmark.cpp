/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LiveSLAMWrapper.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "SlamSystem.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "IOWrapper/ROS/ROSOutput3DWrapper.h"
#include "IOWrapper/ROS/rosReconfigure.h"
#include "DataStructures/FramePoseStruct.h"

#include "util/Undistorter.h"
#include <ros/package.h>

#include "opencv2/opencv.hpp"

namespace lsd_slam {

void runExperiment()
{
  // ---------------------------------------------------------------------------
  // Config
  bool do_slam;
  ros::param::get("~do_slam", do_slam);
  const bool block_until_finished_processing = true;

  // ---------------------------------------------------------------------------
  // Setup

  // get camera calibration in form of an undistorter object.
  // if no undistortion is required, the undistorter will just pass images through.
  std::string dataset_dir = ros::package::getPath("rpg_datasets");
  std::string dataset_name, trace_dir;
  ros::param::get("~dataset_name", dataset_name);
  ros::param::get("~trace_dir", trace_dir);
  std::string calib_file = dataset_dir + "/" + dataset_name + "/lsd_calib.txt";
  Undistorter* undistorter = Undistorter::getUndistorterForFile(calib_file.c_str());
  CHECK_NOTNULL(undistorter);

  int w = undistorter->getOutputWidth();
  int h = undistorter->getOutputHeight();
  int w_inp = undistorter->getInputWidth();
  int h_inp = undistorter->getInputHeight();
  float fx = undistorter->getK().at<double>(0, 0);
  float fy = undistorter->getK().at<double>(1, 1);
  float cx = undistorter->getK().at<double>(2, 0);
  float cy = undistorter->getK().at<double>(2, 1);
  Sophus::Matrix3f K;
  K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

  // make output wrapper. just set to zero if no output is required.
  Output3DWrapper* outputWrapper = new ROSOutput3DWrapper(w,h);

  // make slam system
  SlamSystem* system = new SlamSystem(w, h, K, do_slam);
  system->setVisualization(outputWrapper);
  std::cout << "Setup system complete" << std::endl;

  //--------------------------------------------------------------------------
  // Load dataset
  std::string img_filenames = dataset_dir + "/" + dataset_name + "/data/images.txt";
  std::ifstream img_fs(img_filenames.c_str());
  if(!img_fs.is_open())
  {
    std::cout << "Could not open images file " << img_filenames << std::endl;
    return;
  }

  cv::Mat image = cv::Mat(h,w,CV_8U);
  int first_frame_id, last_frame_id;
  ros::param::get("~dataset_first_frame", first_frame_id);
  ros::param::get("~dataset_last_frame", last_frame_id);
  bool is_initialized = false;
  while(img_fs.good() && !img_fs.eof() && ros::ok())
  {
    if(img_fs.peek() == '#') // skip comments
      img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    //--------------------------------------------------------------------------
    // Load image
    size_t img_id;
    double stamp_seconds;
    std::string img_name;
    img_fs >> img_id >> stamp_seconds >> img_name;
    img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if(img_id < first_frame_id)
      continue;
    else if(img_id > last_frame_id)
      break;
    std::string img_filename(dataset_dir + "/" + dataset_name + "/data/" + img_name);
    cv::Mat imageDist = cv::imread(img_filename, 0);
    CHECK(!imageDist.empty()) << "Image empty.";
    CHECK(imageDist.type() == CV_8U) << "Image not 8uC1";

    if(imageDist.rows != h_inp || imageDist.cols != w_inp)
    {
      printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
               img_name.c_str(), w, h, imageDist.cols, imageDist.rows);
      continue;
    }

    //--------------------------------------------------------------------------
    // Track pose
    undistorter->undistort(imageDist, image);
    CHECK(image.type() == CV_8U) << "Image not 8uC1";
    if(!is_initialized)
    {
      std::cout << "Initialize tracker with frame " << img_id << std::endl;
      system->randomInit(image.data, stamp_seconds, img_id);
      is_initialized = true;
    }
    else
    {
      std::cout << "process frame " << img_id << std::endl;
      system->trackFrame(image.data, img_id, block_until_finished_processing, stamp_seconds);
    }
  }

  // ---------------------------------------------------------------------------
  // Trace poses to file
  std::string trace_pose_filename = trace_dir + "/traj_estimate.txt";
  std::ofstream trace_pose(trace_pose_filename);
  CHECK(!trace_pose.fail());
  const std::vector<FramePoseStruct*, Eigen::aligned_allocator<lsd_slam::FramePoseStruct*> > poses =
      system->getAllPoses();
  std::cout << "Saving " << poses.size() << " poses..." << std::endl;
  for(size_t i = 0; i < poses.size(); ++i)
  {
    FramePoseStruct* pose = poses.at(i);
    const int img_id = pose->frameID;
    const Sim3 T_wc = pose->getCamToWorld();
    const Eigen::Quaterniond& q = T_wc.quaternion();
    const Eigen::Vector3d& p = T_wc.translation();
    trace_pose << img_id << " "
               << p.x() << " " << p.y() << " " << p.z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
               << std::endl;
  }
  std::cout << "...done." << std::endl;

  // ---------------------------------------------------------------------------
  // Finish
  std::cout << "Finished processing." << std::endl;
  //system->finalize(); TODO: This is always blocking!
  delete system;
  delete undistorter;
  delete outputWrapper;
}

} // namespace lsd_slam

int main( int argc, char** argv )
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  ros::init(argc, argv, "LSD_SLAM");

  dynamic_reconfigure::Server<lsd_slam_core::LSDParamsConfig> srv(ros::NodeHandle("~"));
  srv.setCallback(lsd_slam::dynConfCb);

  dynamic_reconfigure::Server<lsd_slam_core::LSDDebugParamsConfig> srvDebug(ros::NodeHandle("~Debug"));
  srvDebug.setCallback(lsd_slam::dynConfCbDebug);

  lsd_slam::runExperiment();
  return 0;
}
