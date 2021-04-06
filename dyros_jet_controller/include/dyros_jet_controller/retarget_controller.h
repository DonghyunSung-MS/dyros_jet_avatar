#ifndef RETARGET_CONTROLLER_H
#define RETARGET_CONTROLLER_H

#include "dyros_jet_controller/dyros_jet_model.h"
#include "dyros_jet_controller/controller.h"
#include "dyros_jet_controller/quadraticprogram.h"
#include "math_type_define.h"

#include <Eigen/Geometry>

#include <fstream>
#include <toml.hpp>

#include <hdf5.hpp>
#include <Eigen/Dense>

#include <boost/filesystem.hpp>

#include <thread>
#include <mutex>

/*TODO:
controlbase callback -> retarget set target
HMD, Waist Controller
*/
namespace dyros_jet_controller
{

struct UpperBodyPoses
{
  double arm_length_[2];
  double shoulder_width_;
  
  Eigen::Vector3d chest2shoulder_offset_[2]; //assume rigid offset
  // master
  // tracker id 0 ~ 6,  pelvis, chest, l_elbow, l_hand, r_elbow, r_hand
  Eigen::Isometry3d tracker_poses_init_[6]; 
  Eigen::Isometry3d tracker_poses_raw_[6]; 
  Eigen::Isometry3d tracker_poses_[6];
  Eigen::Isometry3d tracker_poses_prev_[6];

  //estimated sholuder left and right
  Eigen::Isometry3d shoulder_poses_init_[2];
  // Eigen::Isometry3d shoulder_poses_raw_[2];
  Eigen::Isometry3d shoulder_poses_[2];
  Eigen::Isometry3d shoulder_poses_prev_[2];

  //HMD
  Eigen::Isometry3d hmd_poses_init_; 
  Eigen::Isometry3d hmd_poses_raw_; 
  Eigen::Isometry3d hmd_poses_;
  Eigen::Isometry3d hmd_poses_prev_;
};

class RetargetController
{
public:
  //constructor
  static constexpr unsigned int PRIORITY = 32;

  RetargetController(DyrosJetModel& model, const VectorQd& current_q, 
                     const VectorQd& current_q_dot, const double hz, const double& control_time) :
    total_dof_(DyrosJetModel::HW_TOTAL_DOF), model_(model),
    current_q_(current_q), current_q_dot_(current_q_dot), hz_(hz), control_time_(control_time) 
  {
    //CONTROL FLAG
    ros::param::set("/retarget/control_flag", false);
    ros::param::set("/retarget/initpose_flag", false);
    ros::param::set("/retarget/save_flag", false);
    // ros::param::set("/retarget/still_pose_flag", false);


    //URDF
    std::string desc_package_path = ros::package::getPath("dyros_jet_description");
    std::string urdf_path = desc_package_path + "/robots/dyros_jet_robot.urdf";
    ROS_INFO("RETARGET:: Loading DYROS JET description from = %s",urdf_path.c_str());
    RigidBodyDynamics::Addons::URDFReadFromFile(urdf_path.c_str(), &model_desired_, true, false);

    //QP PARAMETERS
    std::string cont_package_path = ros::package::getPath("dyros_jet_controller");
    std::string toml_path = cont_package_path + "/retarget_config.toml";
    parseToml(toml_path);
    Kp_task_.resize(6,1);
    Kp_task_ << trans_gain_, trans_gain_, trans_gain_, rot_gain_, rot_gain_, rot_gain_;//, elbow_gain_;

    qp_arms_[0].InitializeProblemSize(num_joint_, num_task_);
    qp_arms_[1].InitializeProblemSize(num_joint_, num_task_);
    

    //init pose to model q
    //Arms
    desired_q_ = current_q_;
    desired_q_dot_ = current_q_dot_;

    desired_q_virtual_.setZero();
    desired_q_dot_virtual_.setZero();

    desired_q_virtual_.segment<28>(6) = desired_q_.segment<28>(0);
    desired_q_dot_virtual_.segment<28>(6) = desired_q_dot_.segment<28>(0);


    updateKinematics(desired_q_virtual_, desired_q_dot_virtual_);

    updateArmJacobianFromDesired();

    //data save
    boost::filesystem::path dir(file_path_);
    if(boost::filesystem::create_directories(dir)) {
      std::cout << "Data Path Created" << "\n";
    }

    hf_ = std::make_shared<HDF5::File>(file_path_ + file_name_, HDF5::File::Truncate);


  }
  // base class overide member function
  void compute();
  void writeDesired(const unsigned int *mask, VectorQd& desired_q);
  //tracker value
  void setTrackerTarget(Eigen::Isometry3d T, unsigned int tracker_id);
  void setHMDTarget(Eigen::Isometry3d T);
  //tracker status
  void setPoseCalibrationStatus(int mode); //1: attention, 2:T pose, 3:forward dress. 4:reset
  void setTrackerStatus(bool mode);


private:
  //Retarget joint MAP
  //Tracker id | 0 | 1 | 2 | 3 | 4 | 5 |HMD|
  //joint id   | 12 | . |17 |20 |24 |27 | |
  //joint name | WaistYaw | . |L_ElbowRoll |L_HandYaw |R_ElbowRoll |R_HandYaw |Head|
  unsigned int joint_start_id_[4] = {14, 14, 21, 21};
  // unsigned int arm_joint_id[4] = {16, 20, 23, 27};
  // unsigned int joint_start_id_[4] = {13, 13, 20, 20};
  // unsigned int arm_joint_id[4] = {15, 19, 22, 26};
  
  // const char* arm_joint_name_[4] = {"L_ShoulderYaw_Link", "L_HandYaw_Link", "R_ShoulderYaw_Link", "R_HandYaw_Link"};
  const char* arm_joint_name_[4] = {"L_ElbowRoll_Link", "L_HandYaw_Link", "R_ElbowRoll_Link", "R_HandYaw_Link"};
  const char* tracker_rjoint_name_[6] = {"base_link", "WaistYaw_Link", "L_ShoulderYaw_Link", "L_HandYaw_Link", "R_ShoulderYaw_Link", "R_HandYaw_Link"};
  const char* shoulder_rjoint_name_[2] = {"L_ShoulderPitch_Link", "R_ShoulderPitch_Link"};
  const char* calib_mode_[6] = {"Run_Mode", "Attention", "T", "Forward Dress", "RESET Calibration", "Waiting Calibration"};

  std::shared_ptr<HDF5::File> hf_;
  std::string file_path_, file_name_;

  //stream transform;
  std::vector<Eigen::Matrix<double, 3, 16>> stream_T_current_, stream_T_desired_, stream_T_command_;

  std::mutex mtx_lock_;
  std::thread data_thread_;

  //method
  void parseToml(std::string &toml_path);
  void armJacobianControl();
  void updateArmJacobianFromDesired();
  void getArmJacobian(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian);
  void getArmJacobian(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian, Eigen::Vector3d local_offset);
  void getTransformFromID(unsigned int id, Eigen::Isometry3d &transform_matrix);
  void updateKinematics(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot);
  void setInitRobotPose();
  void getBasisPosition();
  void setStillPose();
  void setMasterScale();
  void poseCalibration();
  void mapping();
  void processRobotMotion();
  void updataMotionData();
  Eigen::VectorXd QPIKArm(unsigned int id);//0 left ,1 right
  void waistControl();
  void logging();

  // Member Variable
  // model info
  unsigned int total_dof_;
  DyrosJetModel &model_;
  RigidBodyDynamics::Model model_desired_;
  Eigen::Matrix<double, 34, 1> desired_q_virtual_;
  Eigen::Matrix<double, 34, 1> desired_q_dot_virtual_;

  const double hz_;
  const double &control_time_; // updated by control_base
  double still_pose_start_time_;
  double still_pose_control_time_;
  double retarget_start_time_;
  double retarget_warmup_time_;
  UpperBodyPoses master_;
  UpperBodyPoses slave_;

  Eigen::Vector6d calib_basis_position_[3]; //Atten(still) , T, forward dress

  //arm jacobians
  Eigen::Matrix<double, 6, 7> jac_hand_[4];
  Eigen::Matrix<double, 7, 6> jac_hand_pinv_[4];
  Eigen::Matrix<double, 7, 7> null_[4];
  int mode_ = 5;

  VectorQd desired_q_;
  VectorQd desired_q_dot_;
  
  const VectorQd &current_q_;  // updated by control_base
  const VectorQd &current_q_dot_;  // updated by control_base

  //Retarget Modes
  bool control_flag_ = false;
  bool initpose_flag_ = false;

  bool still_pose_start_flag_ = false;
  bool still_pose_flag_ = false;
  bool save_flag_ = false;
  bool check_pose_calibration_[5] = {false, false, false, false, false};
  bool real_exp_flag_ = false;
  bool tracker_status_ = false;
  bool retarget_is_first_ = true;

  //QP Variable
  CQuadraticProgram qp_arms_[2];
  const int num_joint_ = 7;
  const int num_task_ = 6;

  Eigen::VectorXd joint_limit_lower_;
  Eigen::VectorXd joint_limit_upper_;

  Eigen::VectorXd joint_vel_limit_lower_;
  Eigen::VectorXd joint_vel_limit_upper_;

  Eigen::VectorXd task_vel_limit_lower_;
  Eigen::VectorXd task_vel_limit_upper_;

  double hand_coeff_;
  double elbow_coeff_;
  double still_criteria_;
  
  double speed_reduce_rate = 20;

  Eigen::VectorXd Kp_task_;

  double rot_gain_;
  double trans_gain_;
  double elbow_gain_;

  double cutoff_freq_;

  Eigen::Vector3d tracker_offset_;

  Eigen::Isometry3d robot_still_pose_[2];

};

} // namespace dyros_jet_controller

#endif // RETARGET_CONTROLLER_H

