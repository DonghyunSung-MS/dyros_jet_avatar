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
  ros::param::get("/retarget/control_flag", control_flag);
  ros::param::get("/retarget/initpose_flag", initpose_flag);
  ros::param::get("/retarget/save_flag", save_flag);


  if(!control_flag)
  {
    setInitPoseArm();
    // data_thread = std::thread(&RetargetController::logging, this);
    // data_thread.detach();
  }
  
  updateKinematics(desired_q_virtual, desired_q_dot_virtual);

  if (control_flag && initpose_flag)
    computeRetarget();
    logging();


  // std::cout<<"diff q "<<(desired_q_.segment<14>(joint_start_id_[0]) - current_q_.segment<14>(joint_start_id_[0])).norm()<<std::endl;
  // std::cout<<"\n\n";
}

void RetargetController::logging()
{
  mtx_lock.lock();

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
    getTransformFromID(model_desired_.GetBodyId(arm_joint_name[i]), T1);
    T_temp<<T1.linear(), T1.translation();
    T_desired.block(0, i*4, 3, 4) = T_temp;

    T_temp.setZero();
    T_temp<<master_pose_arm_poses_[i].linear(), master_pose_arm_poses_[i].translation();
    T_command.block(0, i*4, 3, 4) = T_temp;

    T_temp.setZero();
    model_.getTransformFromID(model_desired_.GetBodyId(arm_joint_name[i]) - 6, T2);
    T_temp<<T2.linear(), T2.translation();
    T_current.block(0, i*4, 3, 4) = T_temp;
  }

  stream_T_current.push_back(T_current);
  stream_T_desired.push_back(T_desired);
  stream_T_command.push_back(T_command);

  //saving
  if (save_flag)
  {
    std::cout<<"saved"<<std::endl;
    Eigen::MatrixXd V(3*stream_T_current.size(), 16);

    V.setZero();
    for(int i=0;i<stream_T_current.size();i++)
    {
        V.block(i*3, 0, 3, 16) = stream_T_current.at(i);
    }
    hf->write("current", V);

    V.setZero();
    for(int i=0;i<stream_T_current.size();i++)
    {
        V.block(i*3, 0, 3, 16) = stream_T_desired.at(i);
    }
    hf->write("desired", V);

    V.setZero();
    for(int i=0;i<stream_T_current.size();i++)
    {
        V.block(i*3, 0, 3, 16) = stream_T_command.at(i);
    }
    hf->write("command", V);
    ros::param::set("/retarget/save_flag", false);

  }
  mtx_lock.unlock();
}

void RetargetController::writeDesired(const unsigned int *mask, VectorQd& desired_q)
{
  for(unsigned int i=11; i<total_dof_; i++)
  {
    
    if(control_flag)
    {
      
      desired_q(i) = desired_q_(i);
    }
  }
}

void RetargetController::computeRetarget()
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
  const double inv_damping = -1e-8;
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


void RetargetController::setArmTarget(Eigen::Isometry3d T, unsigned int tracker_id)
{
  master_pose_arm_poses_prev_[tracker_id - 2] = master_pose_arm_poses_[tracker_id - 2];

  //translation
  master_pose_arm_poses_[tracker_id - 2].translation() = T.translation() + master_pose_arm_poses_init_[tracker_id - 2].translation();
  master_pose_arm_poses_[tracker_id - 2].linear() = T.linear();
}

void RetargetController::getArmJacobian
(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian)
{
  Eigen::MatrixXd full_jacobian(6, DyrosJetModel::MODEL_WITH_VIRTUAL_DOF);
  full_jacobian.setZero();

  RigidBodyDynamics::CalcPointJacobian6D(model_desired_, desired_q_virtual, 
                                         model_desired_.GetBodyId(arm_joint_name[id]),
                                         Eigen::Vector3d::Zero(), full_jacobian, false);


  // std::cout<<full_jacobian<<std::endl;
  jacobian.block<3, 7>(0, 0) = full_jacobian.block<3, 7>(3, joint_start_id_[id] + 6);
  jacobian.block<3, 7>(3, 0) = full_jacobian.block<3, 7>(0, joint_start_id_[id] + 6); 

}

void RetargetController::updateKinematics(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot)
{
  desired_q_virtual.segment<28>(6) = desired_q_.segment<28>(0);
  desired_q_dot_virtual.segment<28>(6) = desired_q_dot_.segment<28>(0);
  RigidBodyDynamics::UpdateKinematicsCustom(model_desired_, &q, &qdot, NULL);
}

void RetargetController::getTransformFromID(unsigned int id, Eigen::Isometry3d &transform_matrix)
{
  transform_matrix.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates
      (model_desired_, desired_q_virtual, id, Eigen::Vector3d::Zero(), false);
  transform_matrix.linear() = RigidBodyDynamics::CalcBodyWorldOrientation
      (model_desired_, desired_q_virtual, id, false).transpose();
}

void RetargetController::setInitPoseArm()
{
  ROS_INFO("Pose initialized!");
  desired_q_ = current_q_;

  //init pose setting
  for (size_t i=0; i<4; i++)
  { 
    Eigen::Isometry3d T_we;
    getTransformFromID(model_desired_.GetBodyId(arm_joint_name[i]), T_we);

    master_pose_arm_poses_init_[i] = T_we;

    master_pose_arm_poses_[i] = master_pose_arm_poses_init_[i];
    master_pose_arm_poses_prev_[i] = master_pose_arm_poses_init_[i];
  }
  
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
  getTransformFromID(model_desired_.GetBodyId(arm_joint_name[elbow_ind]), T_welbow);
  getTransformFromID(model_desired_.GetBodyId(arm_joint_name[hand_ind]), T_whand);
    
  Eigen::Vector6d hand_vel, elbow_vel;
  //hand velocity
  Eigen::VectorXd w_error = -DyrosMath::getPhi(T_whand.linear(), master_pose_arm_poses_[hand_ind].linear());
  Eigen::VectorXd v_error = (master_pose_arm_poses_[hand_ind].translation() - T_whand.translation());

  hand_vel << v_error, w_error;
  hand_vel = Kp_task_.asDiagonal() * hand_vel;

  //elbow velocity
  w_error = -DyrosMath::getPhi(T_welbow.linear(), master_pose_arm_poses_[elbow_ind].linear());
  v_error.setZero();
  elbow_vel << v_error, w_error;
  elbow_vel = Kp_task_.asDiagonal() * elbow_vel;

  Hess = hand_coeff_ * jac_hand_[hand_ind].transpose() * jac_hand_[hand_ind] + \
         elbow_coeff_ * null_[hand_ind].transpose()*jac_hand_[elbow_ind].transpose() * jac_hand_[elbow_ind] * null_[hand_ind];

  grad = - hand_coeff_ * jac_hand_[hand_ind].transpose() * hand_vel + \
         - elbow_coeff_ * null_[hand_ind].transpose()*jac_hand_[elbow_ind].transpose()*(elbow_vel - jac_hand_[elbow_ind] * jac_hand_pinv_[hand_ind] * hand_vel);
  
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

  Eigen::VectorXd qpres, q_dot_solution;
	if(qp_arms_[id].SolveQPoases(100, qpres))
	{
		q_dot_solution = qpres.segment(0, num_joint_);
	}
	else
	{
		q_dot_solution.setZero(num_joint_);
	}

  return q_dot_solution;

}

void RetargetController::parseToml(std::string &toml_path)
{
  auto data = toml::parse(toml_path);

  hand_coeff_ = toml::find<double>(data, "hand_coeff");
  elbow_coeff_ = toml::find<double>(data, "elbow_coeff");

  rot_gain_ = toml::find<double>(data, "rot_gain");
  trans_gain_ = toml::find<double>(data, "trans_gain");

  std::vector<double> tmp = toml::find<std::vector<double>>(data, "joint_limit_lower");
  joint_limit_lower_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(data, "joint_limit_upper");
  joint_limit_upper_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());
  
  tmp = toml::find<std::vector<double>>(data, "joint_vel_limit_lower");
  joint_vel_limit_lower_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(data, "joint_vel_limit_upper");
  joint_vel_limit_upper_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(data, "task_vel_limit_lower");
  task_vel_limit_lower_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  tmp = toml::find<std::vector<double>>(data, "task_vel_limit_upper");
  task_vel_limit_upper_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp.data(), tmp.size());

  file_path = toml::find<std::string>(data, "data_file_path");
  file_name = toml::find<std::string>(data, "data_file_name");
}


}
