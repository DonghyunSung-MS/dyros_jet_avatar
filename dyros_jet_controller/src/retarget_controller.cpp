#include "dyros_jet_controller/retarget_controller.h"
#include "dyros_jet_controller/dyros_jet_model.h"

/*
TODO
  init pose start flag mechanism
*/
namespace dyros_jet_controller
{

void RetargetController::compute()
{
  ros::param::get("/retarget/control_flag", control_flag_);
  ros::param::get("/retarget/initpose_flag", initpose_flag_);
  ros::param::get("/retarget/save_flag", save_flag_);
  // ros::param::get("/retarget/still_pose_flag", still_pose_flag_);

  poseCalibration(); //human motion calibration
  updateKinematics(desired_q_virtual_, desired_q_dot_virtual_);
  //go to still pose
  if(!initpose_flag_)
  {
    setInitRobotPose();
    // data_thread = std::thread(&RetargetController::logging, this);
    // data_thread.detach();
  }
  
  if(initpose_flag_ && !still_pose_flag_)
  {
    if (!still_pose_start_flag_)
    {
      still_pose_start_flag_ = true;
      still_pose_start_time_ = control_time_;
    }

    setStillPose();
    armJacobianControl();
    if (control_time_ > still_pose_start_time_ + still_pose_control_time_)
    {
      still_pose_flag_ = true;
      // ros::param::set("/retarget/still_pose_flag", true);
      setInitRobotPose();

    }
  }


  if (control_flag_ && initpose_flag_)
  {
    if (retarget_is_first_)
    {
      ROS_INFO("START RETARGET CONTROL!");
      retarget_is_first_ = false;
      retarget_start_time_ = control_time_;
      for (int i=1; i < 6; i++)
      {
        master_.tracker_poses_init_[i] = master_.tracker_poses_[i];
        master_.tracker_poses_prev_[i] = master_.tracker_poses_[i];
        
      }
      master_.shoulder_poses_init_[0] = master_.shoulder_poses_[0];
      master_.shoulder_poses_init_[1] = master_.shoulder_poses_[1];

      master_.shoulder_poses_prev_[0] = master_.shoulder_poses_[0];
      master_.shoulder_poses_prev_[1] = master_.shoulder_poses_[1];
      
      std::cout<<"done_init\n";
    }
    mapping();
    processRobotMotion();
    armJacobianControl();
    waistControl();
    updataMotionData();
    logging();
  }
  // std::cout<<"diff q "<<(desired_q_.segment<14>(joint_start_id_[0]) - current_q_.segment<14>(joint_start_id_[0])).norm()<<std::endl;
  // std::cout<<"\n\n";
}

void RetargetController::writeDesired(const unsigned int *mask, VectorQd& desired_q)
{
  for(unsigned int i=12; i<total_dof_; i++)
  {
    
    if(initpose_flag_)
    {
      desired_q(i) = desired_q_(i);
    }
  }
}

void RetargetController::waistControl()
{
  unsigned int waist_ind = 12;
  double q_waist_yaw = desired_q_(waist_ind);

  double q_dot_waist_desired = 0.0;

  // Eigen::AngleAxisd ang_axis(slave_.tracker_poses_[1].linear());
  double error = 0.0;
  error = (slave_.tracker_poses_[1].linear().eulerAngles(0, 1, 2)(2) - q_waist_yaw);
  q_dot_waist_desired = rot_gain_ * error;
  
  std::cout<<q_dot_waist_desired<<std::endl;

  desired_q_dot_(waist_ind) = q_dot_waist_desired;
  desired_q_(waist_ind) = q_dot_waist_desired/hz_ + q_waist_yaw;
}

void RetargetController::armJacobianControl()
{
  //calculate jac and pinv jac
  RetargetController::updateArmJacobianFromDesired();
  Eigen::VectorXd q_arms = desired_q_.segment<14>(joint_start_id_[0]); //left to right

  Eigen::VectorXd q_dot_arms_desired(14);
  q_dot_arms_desired.setZero();

  q_dot_arms_desired.head(7) = QPIKArm(0);
  q_dot_arms_desired.tail(7) = QPIKArm(1);
  
  desired_q_dot_.segment<14>(joint_start_id_[0]) = q_dot_arms_desired;
  desired_q_.segment<14>(joint_start_id_[0]) = q_dot_arms_desired/hz_ + q_arms;
}

void RetargetController::updateArmJacobianFromDesired()
{
  const double inv_damping = -1e-3;
  for (int i=0; i<4; i++)
  {
    Eigen::Matrix<double, 6, 7> jacobian_temp;
    jacobian_temp.setZero();
    getArmJacobian(i, jacobian_temp);

    jac_hand_[i] = jacobian_temp;
    // J.T *(w*I + J*J.T)^-1
    jac_hand_pinv_[i] = jac_hand_[i].transpose() * \
                        (jac_hand_[i] * jac_hand_[i].transpose() + inv_damping * Eigen::MatrixXd::Identity(6, 6)).inverse();

    null_[i] = Eigen::MatrixXd::Identity(7, 7) - jac_hand_pinv_[i]*jac_hand_[i];
  }
}

void RetargetController::setTrackerTarget(Eigen::Isometry3d T, unsigned int tracker_id)
{
  //translation
  if (retarget_is_first_)
    std::cout <<"Populate Raw data "<<tracker_id<<std::endl;
    // std::cout<<"Tracker: "<< T.translation().transpose()<<std::endl;
    // std::cout<<"Tracker: "<< T.linear()<<std::endl;
  master_.tracker_poses_raw_[tracker_id] = T;
}

void RetargetController::setHMDTarget(Eigen::Isometry3d T)
{
  master_.hmd_poses_raw_ = T;
}

void RetargetController::setPoseCalibrationStatus(int mode)
{
  mode_ = mode;
  ROS_INFO("Current Calibration Mode is %s", calib_mode_[mode]);
  if (mode == 4) //reset
  {
    for(int i = 0; i < 5; i ++) check_pose_calibration_[i] = false;
  }

}

void RetargetController::setTrackerStatus(bool mode)
{
  tracker_status_ = mode;
  if(!tracker_status_)
  {
    ROS_INFO("RETARGET::Tracker is not Working!");
  }
}

void RetargetController::getArmJacobian
(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian)
{
  Eigen::MatrixXd full_jacobian(6, DyrosJetModel::MODEL_WITH_VIRTUAL_DOF);
  full_jacobian.setZero();

  RigidBodyDynamics::CalcPointJacobian6D(model_desired_, desired_q_virtual_, 
                                         model_desired_.GetBodyId(arm_joint_name_[id]),
                                         Eigen::Vector3d::Zero(), full_jacobian, false);


  // std::cout<<full_jacobian<<std::endl;
  jacobian.block<3, 7>(0, 0) = full_jacobian.block<3, 7>(3, joint_start_id_[id] + 6);
  jacobian.block<3, 7>(3, 0) = full_jacobian.block<3, 7>(0, joint_start_id_[id] + 6); 

}

void RetargetController::getArmJacobian
(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian, Eigen::Vector3d local_offset)
{
  Eigen::MatrixXd full_jacobian(6, DyrosJetModel::MODEL_WITH_VIRTUAL_DOF);
  full_jacobian.setZero();

  RigidBodyDynamics::CalcPointJacobian6D(model_desired_, desired_q_virtual_, 
                                         model_desired_.GetBodyId(arm_joint_name_[id]),
                                         local_offset, full_jacobian, false);


  // std::cout<<full_jacobian<<std::endl;
  jacobian.block<3, 7>(0, 0) = full_jacobian.block<3, 7>(3, joint_start_id_[id] + 6);
  jacobian.block<3, 7>(3, 0) = full_jacobian.block<3, 7>(0, joint_start_id_[id] + 6); 

}

void RetargetController::updateKinematics(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot)
{
  desired_q_virtual_.segment<28>(6) = desired_q_.segment<28>(0);
  desired_q_dot_virtual_.segment<28>(6) = desired_q_dot_.segment<28>(0);
  RigidBodyDynamics::UpdateKinematicsCustom(model_desired_, &q, &qdot, NULL);
}

void RetargetController::getTransformFromID(unsigned int id, Eigen::Isometry3d &transform_matrix)
{
  transform_matrix.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates
      (model_desired_, desired_q_virtual_, id, Eigen::Vector3d::Zero(), false);
  transform_matrix.linear() = RigidBodyDynamics::CalcBodyWorldOrientation
      (model_desired_, desired_q_virtual_, id, false).transpose();
}

void RetargetController::setInitRobotPose()
{
  ROS_INFO("RETARGET::Pose initialized!");
  desired_q_ = current_q_;

  //init pose setting
  for (size_t i=0; i<6; i++)
  { 
    Eigen::Isometry3d T_we;
    getTransformFromID(model_desired_.GetBodyId(tracker_rjoint_name_[i]), T_we);

    slave_.tracker_poses_init_[i] = T_we;
    slave_.tracker_poses_raw_[i] = T_we;
    slave_.tracker_poses_[i] = T_we;
    slave_.tracker_poses_prev_[i] = T_we;
    // if (i==3)
    //   std::cout << "LEFT_EE : "<<T_we.translation().transpose() <<"\n"<<T_we.linear()<<std::endl;
    // if (i==5)
    //   std::cout << "RIGHT_EE : "<<T_we.translation().transpose() <<"\n"<<T_we.linear()<<std::endl;

  }
  // ROS_INFO("RETARGET::Pose tracker passed!");

  for (size_t i=0; i<2; i++)
  { 
    Eigen::Isometry3d T_we;
    getTransformFromID(model_desired_.GetBodyId(shoulder_rjoint_name_[i]), T_we);

    slave_.shoulder_poses_init_[i] = T_we;
    // slave_.shoulder_poses_raw_[i] = T_we;
    slave_.shoulder_poses_[i] = T_we;
    slave_.shoulder_poses_prev_[i] = T_we;
  }
  // Robot arm length and shoulder width
  // auto T1 = slave_.shoulder_poses_[0].inverse() * slave_.tracker_poses_[3];
  // auto T2 = slave_.shoulder_poses_[1].inverse() * slave_.tracker_poses_[5];

  // Eigen::Vector3d l_arm_l = T1.translation();
  // Eigen::Vector3d r_arm_l = T2.translation();
  // Eigen::Vector3d shoulder_l = slave_.shoulder_poses_[1].translation() - slave_.shoulder_poses_[0].translation();
  // std::cout<<l_arm_l.norm()<<"\t"<<r_arm_l.norm()<<shoulder_l.norm()<<std::endl;
    
  // ROS_INFO("RETARGET::Pose shoulder passed!");

}

Eigen::VectorXd RetargetController::QPIKArm(unsigned int id)
{
  //id: 0 -> jac: 0,1 
  //id: 1 -> jac: 2,3
  unsigned int elbow_ind = id*2;
  unsigned int hand_ind = id*2 + 1;
  
  Eigen::MatrixXd Hess, A;
  Eigen::VectorXd grad, lb, ub, lbA, ubA;

  Hess.setZero(num_joint_, num_joint_);
  grad.setZero(num_joint_);

  lb.setZero(num_joint_);
  ub.setZero(num_joint_);

  lbA.setZero(num_task_);
  ubA.setZero(num_task_);

  Eigen::Isometry3d T_welbow, T_whand;
  getTransformFromID(model_desired_.GetBodyId(tracker_rjoint_name_[elbow_ind + 2]), T_welbow);
  getTransformFromID(model_desired_.GetBodyId(tracker_rjoint_name_[hand_ind + 2]), T_whand);
    
  Eigen::Vector6d hand_vel, elbow_vel;
  //hand velocity
  Eigen::VectorXd w_error = -DyrosMath::getPhi(T_whand.linear(), slave_.tracker_poses_[hand_ind + 2].linear());
  // w_error.setZero();
  Eigen::VectorXd v_error = (slave_.tracker_poses_[hand_ind + 2].translation() - T_whand.translation());

  hand_vel << v_error, w_error;
  hand_vel = Kp_task_.asDiagonal() * hand_vel;

  //elbow velocity
  w_error = -DyrosMath::getPhi(T_welbow.linear(), slave_.tracker_poses_[elbow_ind + 2].linear());
  v_error.setZero();
  elbow_vel << v_error, w_error;
  elbow_vel = Kp_task_.asDiagonal() * elbow_vel;

  w_error(1, 0) = 0.0; //body y 

  Eigen::Matrix<double, 9, 7> J_stack;

  J_stack.block(0, 0, 6, 7) = jac_hand_[hand_ind];
  J_stack.block(6, 0, 3, 7) = jac_hand_[elbow_ind].block(3, 0, 3, 7);

  Hess = J_stack.transpose() * J_stack;
  grad =  - jac_hand_[hand_ind].transpose() * hand_vel - jac_hand_[elbow_ind].block(3, 0, 3, 7).transpose()*w_error;

  // Hess = hand_coeff_ * jac_hand_[hand_ind].transpose() * jac_hand_[hand_ind] + \
  //        elbow_coeff_ * null_[hand_ind].transpose()*jac_hand_[elbow_ind].transpose() * jac_hand_[elbow_ind] * null_[hand_ind];

  // grad = - hand_coeff_ * jac_hand_[hand_ind].transpose() * hand_vel
  //        - elbow_coeff_ * null_[hand_ind].transpose()*jac_hand_[elbow_ind].transpose()*(elbow_vel - jac_hand_[elbow_ind] * desired_q_dot_.segment<7>(joint_start_id_[hand_ind]));
  
  A = jac_hand_[hand_ind];
  
  Eigen::VectorXd q = desired_q_.segment<7>(joint_start_id_[hand_ind]);

  if (id == 0)//left
  {
    for (int i=0; i< num_joint_; i++)
    {
      lb(i) = max(speed_reduce_rate*(joint_limit_lower_(i) - q(i)), joint_vel_limit_lower_(i));
      ub(i) = min(speed_reduce_rate*(joint_limit_upper_(i) - q(i)), joint_vel_limit_upper_(i));
    }
  }

  else if (id ==1)//right
  {
    for (int i=0; i< num_joint_; i++)
    {
      lb(i) = max(speed_reduce_rate*(- joint_limit_upper_(i) - q(i)), joint_vel_limit_lower_(i));
      ub(i) = min(speed_reduce_rate*(- joint_limit_lower_(i) - q(i)), joint_vel_limit_upper_(i));
    }
  }


  lbA = task_vel_limit_lower_;
  ubA = task_vel_limit_upper_;
  
  qp_arms_[id].EnableEqualityCondition(0.0001);
  qp_arms_[id].UpdateMinProblem(Hess, grad);
  qp_arms_[id].UpdateSubjectToAx(A, lbA, ubA);
  qp_arms_[id].UpdateSubjectToX(lb, ub);
  // std::cout<<"ee_"<<id<<" error : "<<hand_vel.transpose()<<std::endl;
  // qp_arms_[id].PrintMinProb();

  Eigen::VectorXd qpres, q_dot_solution;
	if(qp_arms_[id].SolveQPoases(100, qpres))
	{
		q_dot_solution = qpres.segment(0, num_joint_);
	}
	else
	{
		q_dot_solution.setZero(num_joint_);
    qp_arms_[id].InitializeProblemSize(num_joint_, num_task_);
	}

  return q_dot_solution;

}

void RetargetController::parseToml(std::string &toml_path)
{
  auto data = toml::parse(toml_path);
  //parsing robot related data
  auto& robot = toml::find(data, "Robot");
  rot_gain_ = toml::find<double>(robot, "rot_gain");
  trans_gain_ = toml::find<double>(robot, "trans_gain");
  elbow_gain_ = toml::find<double>(robot, "elbow_gain");

  double arm_length = toml::find<double>(robot, "arm_length_max");
  slave_.arm_length_[0] = arm_length;
  slave_.arm_length_[1] = arm_length;
  slave_.shoulder_width_ = toml::find<double>(robot, "shoulder_width");

  //parsing qp variable
  auto& qp_var = toml::find(data, "QPVariable");

  hand_coeff_ = toml::find<double>(qp_var, "hand_coeff");
  elbow_coeff_ = toml::find<double>(qp_var, "elbow_coeff");

  std::vector<double> tmp = toml::find<std::vector<double>>(qp_var, "joint_limit_lower");
  joint_limit_lower_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(qp_var, "joint_limit_upper");
  joint_limit_upper_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());
  
  tmp = toml::find<std::vector<double>>(qp_var, "joint_vel_limit_lower");
  joint_vel_limit_lower_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(qp_var, "joint_vel_limit_upper");
  joint_vel_limit_upper_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(qp_var, "task_vel_limit_lower");
  task_vel_limit_lower_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(qp_var, "task_vel_limit_upper");
  task_vel_limit_upper_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  //parsing retarget data
  auto& retarget_data = toml::find(data, "Retarget");

  real_exp_flag_ = toml::find<bool>(retarget_data, "real_exp");
  tmp = toml::find<std::vector<double>>(retarget_data, "attention_position");
  calib_basis_position_[0] = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(retarget_data, "t_position");
  calib_basis_position_[1] = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(retarget_data, "forward_position");
  calib_basis_position_[2] = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());
  

  tmp = toml::find<std::vector<double>>(retarget_data, "tracker_offset");
  tracker_offset_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(retarget_data, "left_still_pose");
  Eigen::MatrixXd T1 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Map(tmp.data());
  robot_still_pose_[0].translation() = T1.block(0, 3, 3, 1);
  robot_still_pose_[0].linear() = T1.block(0, 0, 3, 3);
  
  tmp = toml::find<std::vector<double>>(retarget_data, "right_still_pose");
  Eigen::MatrixXd T2 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>::Map(tmp.data());
  robot_still_pose_[1].translation() = T2.block(0, 3, 3, 1);
  robot_still_pose_[1].linear() = T2.block(0, 0, 3, 3);

  still_pose_control_time_ = toml::find<double>(retarget_data, "still_pose_control_time");
  still_criteria_ = toml::find<double>(retarget_data, "still_criteria");
  cutoff_freq_ = toml::find<double>(retarget_data, "cutoff_freq");
  retarget_warmup_time_ = toml::find<double>(retarget_data, "warm_up_time");
  //parsing experiment data
  auto& exp_paths = toml::find(data, "Exp");
  file_path_ = toml::find<std::string>(exp_paths, "data_file_path");
  file_name_ = toml::find<std::string>(exp_paths, "data_file_name");

  ROS_INFO("RETARGET:: Parsed Toml Config");
}

void RetargetController::getBasisPosition()
{
  if (mode_ > 0 && mode_ < 5)
  {  
    ROS_INFO("GET BASIS POSE FROM TRACKER %d, %d", mode_, check_pose_calibration_[mode_]);
    if (real_exp_flag_)
      calib_basis_position_[mode_] << master_.tracker_poses_[3].translation() - master_.tracker_poses_[1].translation(),\
                                      master_.tracker_poses_[5].translation() - master_.tracker_poses_[1].translation();
    check_pose_calibration_[mode_] = true;
  }
}

void RetargetController::setMasterScale() //Assume 3 basis position is acheive;
{
  /**
   * Estimate shoulder location from chest frame
   * Estimate arm length
   **/
  //simpliest shoulder location estimation
  Eigen::Vector3d lsh_tmp, rsh_tmp;
  lsh_tmp(0) = (calib_basis_position_[0](0) + calib_basis_position_[1](0)) / 2.0; //x_still, x_t
  lsh_tmp(1) = (calib_basis_position_[0](1)); //y_still
  lsh_tmp(2) = (calib_basis_position_[1](2) + calib_basis_position_[2](2)); //z_t, z_forward
  
  rsh_tmp(0) = (calib_basis_position_[0](3) + calib_basis_position_[1](3)) / 2.0; //x_still, x_t
  rsh_tmp(1) = (calib_basis_position_[0](4)); //y_still
  rsh_tmp(2) = (calib_basis_position_[1](5) + calib_basis_position_[2](5)); //z_t, z_forward
  
  //arm length
  double l_arm_len = 0;
  double r_arm_len = 0;

  for (int i=0; i<3; i++)
  {
    l_arm_len += (lsh_tmp - calib_basis_position_[i].head(3)).norm();
    r_arm_len += (rsh_tmp - calib_basis_position_[i].tail(3)).norm();
  }

  l_arm_len /= 3.0;
  r_arm_len /= 3.0;
  
  //shoulder_width
  double sh_width = (lsh_tmp - rsh_tmp).norm();

  master_.chest2shoulder_offset_[0] = lsh_tmp;// + master_.tracker_poses_[1].translation();
  master_.chest2shoulder_offset_[1] = rsh_tmp;// + master_.tracker_poses_[1].translation();

  master_.shoulder_width_ = sh_width;
  master_.arm_length_[0] = l_arm_len;
  master_.arm_length_[1] = r_arm_len;
  
  check_pose_calibration_[0] = true;
}

void RetargetController::poseCalibration()
{
  /**
   * change tracker world frame to base frame
   * use only z rotation for chest(1) tracker
   **/
  //raw data into pelv frame
  for (int i=1; i<6; i++)
  {
    master_.tracker_poses_[i] = master_.tracker_poses_raw_[0].inverse() * master_.tracker_poses_raw_[i];
    // std::cout<<i<<" master tracker post : "<<master_.tracker_poses_[i].translation().transpose()<<std::endl;
  }

  //chest use only z rotation
  Eigen::Vector3d master_pelv_rpy;
	Eigen::Matrix3d master_pelv_yaw_rot;
	master_pelv_rpy = DyrosMath::rot2Euler(master_.tracker_poses_raw_[1].linear());
	master_pelv_yaw_rot = DyrosMath::rotateWithZ(master_pelv_rpy(2));
	master_.tracker_poses_[1].linear() = master_pelv_yaw_rot;

  // master_.tracker_poses_[3].translation() = master_.tracker_poses_[3].linear() * tracker_offset_;
  // master_.tracker_poses_[5].translation() = master_.tracker_poses_[5].linear() * tracker_offset_;
  
  if (!check_pose_calibration_[1] && mode_==1) //Attention
    getBasisPosition();
  else if(!check_pose_calibration_[2] && mode_==2) //T
    getBasisPosition();
  else if(!check_pose_calibration_[3] && mode_==3) //Forward
    getBasisPosition();

  if (!check_pose_calibration_[0] && check_pose_calibration_[1] && check_pose_calibration_[2] && check_pose_calibration_[3])
  {
    ROS_INFO("SET MASTER SCALE");
    setMasterScale();
  }

  for (int i=0; i<2; i++)
  {
    //shoulder position in chest frame
    master_.shoulder_poses_[i].translation() = master_.chest2shoulder_offset_[i];// + master_.tracker_poses_[1].translation();
    master_.shoulder_poses_[i].linear() = master_.tracker_poses_[1].linear();
  }
}

void RetargetController::setStillPose()
{
  //translation
  //left
  Eigen::Isometry3d T_el[2];
  getTransformFromID(model_desired_.GetBodyId(arm_joint_name_[0]), T_el[0]);
  getTransformFromID(model_desired_.GetBodyId(arm_joint_name_[2]), T_el[1]);

  Eigen::Vector3d init_position, final_position, zero_vector;
  zero_vector.setZero();
  init_position = slave_.tracker_poses_init_[3].translation();
  final_position = robot_still_pose_[0].translation();
  slave_.tracker_poses_[2].translation() = T_el[0].translation();
  slave_.tracker_poses_[3].translation() = DyrosMath::cubicVector(control_time_, still_pose_start_time_, still_pose_start_time_ + still_pose_control_time_,
                                                                  init_position, final_position,
                                                                  zero_vector, zero_vector);
  //right
  init_position = slave_.tracker_poses_init_[5].translation();
  final_position = robot_still_pose_[1].translation();
  slave_.tracker_poses_[4].translation() =T_el[1].translation();
  slave_.tracker_poses_[5].translation() = DyrosMath::cubicVector(control_time_, still_pose_start_time_, still_pose_start_time_ + still_pose_control_time_,
                                                                  init_position, final_position,
                                                                  zero_vector, zero_vector);

  //rotation
  slave_.tracker_poses_[2].linear() = slave_.tracker_poses_init_[2].linear();
  slave_.tracker_poses_[3].linear() = DyrosMath::rotationCubic(control_time_, still_pose_start_time_, still_pose_start_time_ + still_pose_control_time_,
                                                                  slave_.tracker_poses_init_[3].linear(), robot_still_pose_[0].linear());
  //right
  slave_.tracker_poses_[4].linear() = slave_.tracker_poses_init_[4].linear();
  slave_.tracker_poses_[5].linear() = DyrosMath::rotationCubic(control_time_, still_pose_start_time_, still_pose_start_time_ + still_pose_control_time_,
                                                                  slave_.tracker_poses_init_[5].linear(), robot_still_pose_[1].linear());
        
}

void RetargetController::mapping()
{
  
  //solvind A_h w = b_h to get h2r position coefficient
  //A_r * w* = b_r, where b_r represetns mapped_robot position
  
  // std::cout<<"is in???";
  Eigen::Matrix<double, 3, 4> A_h[2], A_r[2];
  Eigen::Matrix<double, 3, 1> b_h[2], b_r[2];
  Eigen::MatrixXd A_h_inv[2];

  A_h[0].setZero();
  A_h[1].setZero();

  for (int i=0;i<3;i++)
  {
    A_h[0].block(0, i, 3, 1) = calib_basis_position_[i].head(3) - master_.shoulder_poses_[0].translation();
    A_h[1].block(0, i, 3, 1) = calib_basis_position_[i].tail(3) - master_.shoulder_poses_[1].translation();
    
  }
  A_h[0].block(0, 3, 3, 1) = master_.shoulder_poses_[1].translation() - master_.shoulder_poses_[0].translation();//shoulder
  A_h[1].block(0, 3, 3, 1) = master_.shoulder_poses_[0].translation() - master_.shoulder_poses_[1].translation(); //shoulder

  // //human shoulder2hand
  b_h[0] = master_.tracker_poses_[3].translation() -  master_.tracker_poses_[1].translation() - master_.shoulder_poses_[0].translation();
  b_h[1] = master_.tracker_poses_[5].translation() -  master_.tracker_poses_[1].translation() - master_.shoulder_poses_[1].translation();

  A_r[0].setZero();
  A_r[1].setZero();

  for (int i=0;i<3;i++)
  {
    double tmp1 = i - 1.5;
    double tmp2 = i - 0.5;
    
    A_r[0](i, 2 - i) = - tmp1 / abs(tmp1) * slave_.arm_length_[0]; //++-
    A_r[1](i, 2 - i) = - tmp2 / abs(tmp2) * slave_.arm_length_[1]; //+--
    
  }

  A_r[0](1, 3) = - slave_.shoulder_width_;
  A_r[1](1, 3) =   slave_.shoulder_width_;
  
  // //robot shoulder2hand
  for (int i=0;i<2;i++)
  {
    double mag;
    A_h_inv[i] = A_h[i].transpose() * (A_h[i] * A_h[i].transpose() + 1e-5 * Eigen::MatrixXd::Identity(3, 3)).inverse();
    b_r[i] = A_r[i] * A_h_inv[i] * b_h[i];
    //rescale
    mag = std::max(0.1, std::min(b_r[i].norm(), slave_.arm_length_[i]));
    b_r[i] = mag * b_r[i].normalized();
  }
  // std::cout<<"ROBOT A mat"<<A_r[0]<<A_r[1]<<std::endl;
  // std::cout<<"ROBOT b mat"<<b_r[0].transpose()<<b_r[1].transpose()<<std::endl;
  //-------------------------------------------------------------------------------------FRAME MATCHING----------------------------------------------------------------------------------
  // //robot based2hand
  Eigen::Isometry3d T_b2sh_l, T_b2sh_r;
  getTransformFromID(model_desired_.GetBodyId(shoulder_rjoint_name_[0]), T_b2sh_l);
  getTransformFromID(model_desired_.GetBodyId(shoulder_rjoint_name_[1]), T_b2sh_r);


  slave_.tracker_poses_[3].translation() = T_b2sh_l.translation() + b_r[0];
  slave_.tracker_poses_[5].translation() = T_b2sh_r.translation() + b_r[1]; 
  
  //hand rotation
  slave_.tracker_poses_[3].linear() = master_.tracker_poses_[3].linear() * DyrosMath::rotateWithZ(-M_PI * 0.5) * DyrosMath::rotateWithX(-M_PI * 0.5);
  slave_.tracker_poses_[5].linear() = master_.tracker_poses_[5].linear() * DyrosMath::rotateWithZ(M_PI * 0.5) * DyrosMath::rotateWithX(M_PI * 0.5);
  // slave_.tracker_poses_[3].linear() = master_.tracker_poses_[3].linear();
  // slave_.tracker_poses_[5].linear() = master_.tracker_poses_[5].linear();

  // elbow translation
  b_h[0] = master_.tracker_poses_[2].translation() -  master_.tracker_poses_[1].translation() - master_.shoulder_poses_[0].translation();
  b_h[1] = master_.tracker_poses_[4].translation() -  master_.tracker_poses_[1].translation() - master_.shoulder_poses_[1].translation();

  A_r[0].setZero();
  A_r[1].setZero();

  for (int i=0;i<3;i++)
  {
    double tmp1 = i - 1.5;
    double tmp2 = i - 0.5;
    
    A_r[0](i, 2 - i) = - tmp1 / abs(tmp1) * slave_.arm_length_[0]; //++-
    A_r[1](i, 2 - i) = - tmp2 / abs(tmp2) * slave_.arm_length_[1]; //+--
    
  }

  A_r[0](1, 3) = - slave_.shoulder_width_;
  A_r[1](1, 3) =   slave_.shoulder_width_;
  
  // //robot shoulder2hand
  for (int i=0;i<2;i++)
  {
    double mag;
    A_h_inv[i] = A_h[i].transpose() * (A_h[i] * A_h[i].transpose() + 1e-5 * Eigen::MatrixXd::Identity(3, 3)).inverse();
    b_r[i] = A_r[i] * A_h_inv[i] * b_h[i];
    //rescale
    mag = std::max(0.1, std::min(b_r[i].norm(), slave_.arm_length_[i]/2.0));
    b_r[i] = mag * b_r[i].normalized();
  }
  slave_.tracker_poses_[2].translation() = T_b2sh_l.translation() + b_r[0];
  slave_.tracker_poses_[4].translation() = T_b2sh_r.translation() + b_r[1]; 
  
  slave_.tracker_poses_[2].linear() = slave_.tracker_poses_init_[2].linear() * \
                                      master_.tracker_poses_init_[2].linear().transpose() * master_.tracker_poses_[2].linear();
  slave_.tracker_poses_[4].linear() = slave_.tracker_poses_init_[4].linear() * \
                                      master_.tracker_poses_init_[4].linear().transpose() * master_.tracker_poses_[4].linear();

  // //chest reduce rotation
  Eigen::Matrix3d R_delchest = master_.tracker_poses_init_[1].linear().transpose() * master_.tracker_poses_[1].linear();
  Eigen::AngleAxisd target_axis_angle(R_delchest);
  Eigen::Matrix3d R_filtered;
  R_filtered = Eigen::AngleAxisd(0.5*target_axis_angle.angle(), target_axis_angle.axis());
  slave_.tracker_poses_[1].linear() = slave_.tracker_poses_init_[1].linear() * R_filtered;
}

void RetargetController::processRobotMotion()
{
  // DyrosMath::lpf<3>(in, prev, hz_, cutoff_freq_)
  for (int i = 1;i < 6; i++)
  {
    slave_.tracker_poses_[i].translation() = DyrosMath::lpf<3>(slave_.tracker_poses_[i].translation(), 
                                                              slave_.tracker_poses_prev_[i].translation(),
                                                              hz_, cutoff_freq_);
    //rotation filter
    Eigen::Matrix3d target_R = (slave_.tracker_poses_prev_[i].linear().transpose() * slave_.tracker_poses_[i].linear());
    Eigen::AngleAxisd target_axis_angle(target_R);

    Eigen::Matrix3d R_filtered;
    R_filtered = Eigen::AngleAxisd(DyrosMath::lpf(target_axis_angle.angle(), 0.0, hz_, cutoff_freq_), target_axis_angle.axis());
    slave_.tracker_poses_[i].linear() = slave_.tracker_poses_prev_[i].linear() * R_filtered;
  }

  if(control_time_ < retarget_warmup_time_ + retarget_start_time_) //prevent infeasible velocity
  {
    Eigen::Isometry3d T_desired_model;
    Eigen::Vector3d init_position, final_position, zero_vector;
    zero_vector.setZero();

    for (int i=1;i<6;i++)
    {
      getTransformFromID(model_desired_.GetBodyId(tracker_rjoint_name_[i]), T_desired_model);
      init_position = slave_.tracker_poses_init_[i].translation();
      final_position = slave_.tracker_poses_[i].translation();
      slave_.tracker_poses_[i].translation() = DyrosMath::cubicVector(control_time_, retarget_warmup_time_, retarget_warmup_time_ + retarget_start_time_,
                                                                    init_position, final_position,
                                                                    zero_vector, zero_vector);
      slave_.tracker_poses_[i].linear() = DyrosMath::rotationCubic(control_time_, retarget_warmup_time_, retarget_warmup_time_ + retarget_start_time_,
                                                                   slave_.tracker_poses_init_[i].linear(), slave_.tracker_poses_[i].linear());
    }
  }
}

void RetargetController::updataMotionData()
{
  for(int i=0;i<6;i++)
  {
    master_.tracker_poses_prev_[i] = master_.tracker_poses_[i];
    slave_.tracker_poses_prev_[i] = slave_.tracker_poses_[i];

    if (i==0 || i==1)
    {
      master_.shoulder_poses_prev_[i] = master_.shoulder_poses_[i];
      slave_.shoulder_poses_prev_[i] = slave_.shoulder_poses_[i];
    }
  }
}

void RetargetController::logging()
{
  // mtx_lock_.lock();

  Eigen::Matrix<double, 3, 16> T_current, T_desired, T_command;

  T_current.setZero();
  T_desired.setZero();
  T_command.setZero();

  for(int i=0;i<4;i++)
  {
    //desired
    Eigen::Matrix<double,3,4> T_temp;
    Eigen::Isometry3d T1, T2;

    T_temp.setZero();
    getTransformFromID(model_desired_.GetBodyId(arm_joint_name_[i]), T1);
    T_temp<<T1.linear(), T1.translation();
    T_desired.block(0, i*4, 3, 4) = T_temp;

    T_temp.setZero();
    T_temp<<slave_.tracker_poses_[i+2].linear(), slave_.tracker_poses_[i+2].translation();
    T_command.block(0, i*4, 3, 4) = T_temp;

    T_temp.setZero();
    model_.getTransformFromID(model_desired_.GetBodyId(arm_joint_name_[i]) - 6, T2);
    T_temp<<T2.linear(), T2.translation();
    T_current.block(0, i*4, 3, 4) = T_temp;
    if (i==1 || i==3)
    {
      Eigen::AngleAxisd ang_axs_master(master_.tracker_poses_[i+2].linear());
      Eigen::AngleAxisd ang_axs_slave(slave_.tracker_poses_[i+2].linear());
      Eigen::AngleAxisd ang_axs_current(T2.linear()); 

      // std::cout<<" \tslave : "<<slave_.tracker_poses_[i+2].translation().transpose()<<" | "<<ang_axs_slave.angle() * ang_axs_slave.axis().transpose()<<std::endl;
      // std::cout<<" \trobot : "<<T2.translation().transpose()<<" | "<<ang_axs_current.angle() * ang_axs_current.axis().transpose()<<std::endl;
    }
  }
  for(int i=1;i<6;i++)
  {
    Eigen::AngleAxisd ang_axs_master(master_.tracker_poses_[i].linear());
    std::cout<<i<<"\tmaster : "<<master_.tracker_poses_[i].translation().transpose()<<" | "<<ang_axs_master.angle() * ang_axs_master.axis().transpose() <<std::endl;
  }

  stream_T_current_.push_back(T_current);
  stream_T_desired_.push_back(T_desired);
  stream_T_command_.push_back(T_command);

  // saving
  if (save_flag_)
  {
    std::cout<<"saved"<<std::endl;
    Eigen::MatrixXd V(3*stream_T_current_.size(), 16);

    V.setZero();
    for(int i=0;i<stream_T_current_.size();i++)
    {
        V.block(i*3, 0, 3, 16) = stream_T_current_.at(i);
    }
    hf_->write("current", V);

    V.setZero();
    for(int i=0;i<stream_T_current_.size();i++)
    {
        V.block(i*3, 0, 3, 16) = stream_T_desired_.at(i);
    }
    hf_->write("desired", V);

    V.setZero();
    for(int i=0;i<stream_T_current_.size();i++)
    {
        V.block(i*3, 0, 3, 16) = stream_T_command_.at(i);
    }
    hf_->write("command", V);
    ros::param::set("/retarget/save_flag", false);

  }
  // mtx_lock_.unlock();
}

}
