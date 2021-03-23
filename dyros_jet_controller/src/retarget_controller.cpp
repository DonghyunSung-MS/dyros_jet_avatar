#include "dyros_jet_controller/retarget_controller.h"
#include "dyros_jet_controller/dyros_jet_model.h"

namespace dyros_jet_controller
{

void RetargetController::compute()
{
  if (start_flag)
  {
    desired_q_ = current_q_;
    start_flag = false;
  }
  RetargetController::computeRetarget();
}

void RetargetController::writeDesired(const unsigned int *mask, VectorQd& desired_q)
{
  for(unsigned int i=11; i<total_dof_; i++)
  {
    
    // if( mask[i] >= PRIORITY && mask[i] < PRIORITY * 2 )
    // {
      
    desired_q(i) = desired_q_(i);
    // }
  }
  // std::cout<<"RETARGET"<<desired_q_.transpose()<<std::endl;
}

void RetargetController::computeRetarget()
{
  //calculate jac and pinv jac
  RetargetController::updateArmJacobianFromDesired();
  Eigen::VectorXd q_arms = desired_q_.segment<14>(joint_start_id_[0]); //left to right

  Eigen::VectorXd q_dot_arms(14);
  q_dot_arms.setZero();
  Eigen::VectorXd Kp_task(6);

  double rot_gain = 20;
  double trans_gain = 200;

  Kp_task << rot_gain, rot_gain, rot_gain, \
             trans_gain, trans_gain, trans_gain;

  for (int i=0; i<4; i++)
  {
    Eigen::Vector6d V_error;
    const auto &x = model_.getCurrentTransform((DyrosJetModel::EndEffector)(int(i/2) + 2));

    Eigen::VectorXd w_error = DyrosMath::getPhi(master_pose_arm_poses_prev_[i].linear(), master_pose_arm_poses_[i].linear());

    Eigen::VectorXd v_error = master_pose_arm_poses_[i].translation() - master_pose_arm_poses_prev_[i].translation();

    // Eigen::VectorXd v_error = master_pose_arm_poses_[i].translation() - x.translation();
    V_error << v_error, w_error;


    if (i == 1)
      q_dot_arms.head(7) = jac_hand_pinv_[i] * Kp_task.asDiagonal() * V_error;
    else if(i == 3)
      q_dot_arms.tail(7) = jac_hand_pinv_[i] * Kp_task.asDiagonal() * V_error;

    std::cout<<i<<std::endl;
    std::cout<<V_error.transpose()<<std::endl;
    std::cout<<x.translation().transpose()<<std::endl;
    std::cout<<q_dot_arms.transpose()<<std::endl;
  }
  std::cout<<"\n\n\n";
  desired_q_.segment<14>(joint_start_id_[0]) = q_dot_arms/hz_ + q_arms;
}

void RetargetController::updateArmJacobianFromDesired()
{
  const double inv_damping = 1e-4;
  for (int i=0; i<4; i++)
  {
    Eigen::Matrix<double, 6, 7> jacobian_temp;
    jacobian_temp.setZero();
    getArmJacobian(i, jacobian_temp);
    std::cout<<"jacobian hand "<<i<<std::endl;
    std::cout<<jacobian_temp<<std::endl;
    jac_hand_[i] = jacobian_temp;
    // J.T *(w*I + J*J.T)^-1
    jac_hand_pinv_[i] = jac_hand_[i].transpose() * \
                        (jac_hand_[i] * jac_hand_[i].transpose() + inv_damping * Eigen::MatrixXd::Identity(7, 7)).inverse();
  }
}


void RetargetController::setArmTarget(Eigen::Isometry3d T, unsigned int tracker_id)
{
  // std::cout<<"ID : "<<tracker_id<<" : \tTranslation"<<T.translation().transpose()<<std::endl;
  master_pose_arm_poses_prev_[tracker_id - 2] = master_pose_arm_poses_[tracker_id - 2];
  master_pose_arm_poses_[tracker_id - 2] = T;
}

void RetargetController::getArmJacobian
(unsigned int id, Eigen::Matrix<double, 6, 7> &jacobian)
{
  Eigen::MatrixXd full_jacobian(6, DyrosJetModel::MODEL_WITH_VIRTUAL_DOF);
  full_jacobian.setZero();

  RigidBodyDynamics::CalcPointJacobian6D(model_desired_, desired_q_, 
                                         model_desired_.GetBodyId(arm_joint_name[id]),
                                         Eigen::Vector3d::Zero(), full_jacobian, true);

  jacobian.block<3, 7>(0, 0) = full_jacobian.block<3, 7>(3, joint_start_id_[id] + 6);
  jacobian.block<3, 7>(3, 0) = full_jacobian.block<3, 7>(0, joint_start_id_[id] + 6);
}


void RetargetController::updateControlMask(unsigned int *mask)
{
  unsigned int index = 0;
  for(int i=0; i<total_dof_; i++)
  {
    if(i < 6)
    {
      index = 0;
    }
    else if (i < 6 + 6)
    {
      index = 1;
    }
    else if (i < 6 + 6 + 2)
    {
      continue; // waist
    }
    else if (i < 6 + 6 + 2 + 7)
    {
      index = 2;
    }
    else if (i < 6 + 6 + 2 + 7 + 7)
    {
      index = 3;
    }

    if(ee_enabled_[index])
    {
      if (mask[i] >= PRIORITY * 2)
      {
        // Higher priority task detected
        ee_enabled_[index] = false;
        target_transform_[index] = model_.getCurrentTransform((DyrosJetModel::EndEffector)index);
        end_time_[index] = control_time_;
        if (index < 2)  // Legs
        {
          desired_q_.segment<6>(model_.joint_start_index_[index]) = current_q_.segment<6>(model_.joint_start_index_[index]);
        }
        else
        {
          desired_q_.segment<7>(model_.joint_start_index_[index]) = current_q_.segment<7>(model_.joint_start_index_[index]);
        }
        //setTarget((DyrosJetModel::EndEffector)index, model_.getCurrentTransform((DyrosJetModel::EndEffector)index), 0); // Stop moving
        target_arrived_[index] = true;
      }
      mask[i] = (mask[i] | PRIORITY);
    }
    else
    {
      mask[i] = (mask[i] & ~PRIORITY);
      //setTarget((DyrosJetModel::EndEffector)index, model_.getCurrentTransform((DyrosJetModel::EndEffector)index), 0); // Stop moving
      target_arrived_[index] = true;
    }
  }
}
} // namespace dyros_jet_controller


