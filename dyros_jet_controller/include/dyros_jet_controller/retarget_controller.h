#ifndef RETARGET_CONTROLLER_H
#define RETARGET_CONTROLLER_H

#include "dyros_jet_controller/dyros_jet_model.h"
#include "dyros_jet_controller/controller.h"
#include "math_type_define.h"
#include <Eigen/Geometry>
#include <fstream>

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
      std::string desc_package_path = ros::package::getPath("dyros_jet_description");
      std::string urdf_path = desc_package_path + "/robots/dyros_jet_robot.urdf";

      ROS_INFO("Loading DYROS JET description from = %s",urdf_path.c_str());
      RigidBodyDynamics::Addons::URDFReadFromFile(urdf_path.c_str(), &model_desired_, true, false);
      //init pose to model q
      //Arms
      desired_q_ = current_q_;
      
      for (size_t i=0; i<4; i++)
      {
        model_.getTransformFromID(model_desired_.GetBodyId(arm_joint_name[i]), master_pose_arm_poses_[i]);
        master_pose_arm_poses_prev_[i] = master_pose_arm_poses_[i];
      }
      updateArmJacobianFromDesired();

    }
  // base class overide member function
  void compute();
  void updateControlMask(unsigned int *mask);
  void writeDesired(const unsigned int *mask, VectorQd& desired_q);
  void setArmTarget(Eigen::Isometry3d T, unsigned int tracker_id);
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
  std::ofstream dfile("desired.txt");
  std::ofstream cfile("current.txt");
  


  //retarget function
  void computeRetarget();
  void updateArmJacobianFromDesired();
  void getArmJacobian(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian);
  // Member Variable
  // model info
  unsigned int total_dof_;
  DyrosJetModel &model_;
  RigidBodyDynamics::Model model_desired_;

  const double hz_;
  const double &control_time_; // updated by control_base
  bool start_flag = true;
  // master
  // Left to Right 

  Eigen::Isometry3d master_pose_arm_poses_[4]; //left to right, (elbow, hand)
  Eigen::Isometry3d master_pose_arm_poses_prev_[4]; //left to right, (elbow, hand)

  // slave(robot)
  Eigen::Matrix<double, 6, 7> jac_hand_[4];
  Eigen::Matrix<double, 7, 6> jac_hand_pinv_[4];

  VectorQd desired_q_;

  const VectorQd &current_q_;  // updated by control_base
  const VectorQd &current_q_dot_;  // updated by control_base

  bool ee_enabled_[4];
  double start_time_[4];
  double end_time_[4];
  bool target_arrived_[4];

  Eigen::Matrix3d rot_init_;
  Eigen::Isometry3d start_transform_[4];
  Eigen::Isometry3d previous_transform_[4];
  Eigen::Isometry3d desired_transform_[4];
  Eigen::Isometry3d target_transform_[4];
};

} // namespace dyros_jet_controller

#endif // RETARGET_CONTROLLER_H

