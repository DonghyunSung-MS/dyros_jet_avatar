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

    //URDF
    std::string desc_package_path = ros::package::getPath("dyros_jet_description");
    std::string urdf_path = desc_package_path + "/robots/dyros_jet_robot.urdf";
    ROS_INFO("Loading DYROS JET description from = %s",urdf_path.c_str());
    RigidBodyDynamics::Addons::URDFReadFromFile(urdf_path.c_str(), &model_desired_, true, false);

    //QP PARAMETERS
    std::string cont_package_path = ros::package::getPath("dyros_jet_controller");
    std::string toml_path = cont_package_path + "/retarget_config.toml";
    parseToml(toml_path);
    Kp_task_ << trans_gain_, trans_gain_, trans_gain_, rot_gain_, rot_gain_, rot_gain_;

    qp_arms_[0].InitializeProblemSize(num_joint_, num_task_);
    qp_arms_[1].InitializeProblemSize(num_joint_, num_task_);
    

    //init pose to model q
    //Arms
    desired_q_ = current_q_;
    desired_q_dot_ = current_q_dot_;

    desired_q_virtual.setZero();
    desired_q_dot_virtual.setZero();

    desired_q_virtual.segment<28>(6) = desired_q_.segment<28>(0);
    desired_q_dot_virtual.segment<28>(6) = desired_q_dot_.segment<28>(0);


    updateKinematics(desired_q_virtual, desired_q_dot_virtual);

    updateArmJacobianFromDesired();

    //data save
    boost::filesystem::path dir(file_path);
    if(boost::filesystem::create_directories(dir)) {
      std::cout << "Data Path Created" << "\n";
    }

    hf = std::make_shared<HDF5::File>(file_path + file_name, HDF5::File::Truncate);


  }
  // base class overide member function
  void compute();
  void writeDesired(const unsigned int *mask, VectorQd& desired_q);
  void setArmTarget(Eigen::Isometry3d T, unsigned int tracker_id);
  void parseToml(std::string &toml_path);
  //Retarget joint MAP
  //Tracker id | 0 | 1 | 2 | 3 | 4 | 5 |HMD|
  //joint id   | 12 | . |17 |20 |24 |27 | |
  //joint name | WaistYaw | . |L_ElbowRoll |L_HandYaw |R_ElbowRoll |R_HandYaw |Head|
  unsigned int joint_start_id_[4] = {14, 14, 21, 21};
  // unsigned int arm_joint_id[4] = {16, 20, 23, 27};
  // unsigned int joint_start_id_[4] = {13, 13, 20, 20};
  // unsigned int arm_joint_id[4] = {15, 19, 22, 26};
  

  // const char* arm_joint_name[4] = {"L_ElbowRoll_Link", "L_HandYaw_Link", "R_ElbowRoll_Link", "R_HandYaw_Link"};
  const char* arm_joint_name[4] = {"L_ShoulderYaw_Link", "L_HandYaw_Link", "R_ShoulderYaw_Link", "R_HandYaw_Link"};

private:
  std::shared_ptr<HDF5::File> hf;
  std::string file_path, file_name;

  //stream transform;
  std::vector<Eigen::Matrix<double, 3, 16>> stream_T_current, stream_T_desired, stream_T_command;

  std::mutex mtx_lock;
  std::thread data_thread;

  //retarget function
  void computeRetarget();
  void updateArmJacobianFromDesired();
  void getArmJacobian(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian);
  void getTransformFromID(unsigned int id, Eigen::Isometry3d &transform_matrix);
  void updateKinematics(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot);
  void setInitPoseArm();
  Eigen::VectorXd QPIKArm(unsigned int id);//0 left ,1 right
  void logging();

  // Member Variable
  // model info
  unsigned int total_dof_;
  DyrosJetModel &model_;
  RigidBodyDynamics::Model model_desired_;
  Eigen::Matrix<double, 34, 1> desired_q_virtual;
  Eigen::Matrix<double, 34, 1> desired_q_dot_virtual;


  const double hz_;
  const double &control_time_; // updated by control_base
  // master
  // Left to Right 
  Eigen::Isometry3d master_pose_arm_poses_init_[4]; //left to right, (elbow, hand)
  Eigen::Isometry3d master_pose_arm_poses_[4]; //left to right, (elbow, hand)
  Eigen::Isometry3d master_pose_arm_poses_prev_[4]; //left to right, (elbow, hand)

  // slave(robot)
  Eigen::Matrix<double, 6, 7> jac_hand_[4];
  Eigen::Matrix<double, 7, 6> jac_hand_pinv_[4];
  Eigen::Matrix<double, 7, 7> null_[4];

  VectorQd desired_q_;
  VectorQd desired_q_dot_;
  

  const VectorQd &current_q_;  // updated by control_base
  const VectorQd &current_q_dot_;  // updated by control_base


  bool control_flag = false;
  bool initpose_flag = false;
  bool save_flag = false;

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
  
  double speed_reduce_rate = 20;

  Eigen::Vector6d Kp_task_;

  double rot_gain_;
  double trans_gain_;

};

} // namespace dyros_jet_controller

#endif // RETARGET_CONTROLLER_H

