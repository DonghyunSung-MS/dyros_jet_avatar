#ifndef WALKING_CONTROLLER_H
#define WALKING_CONTROLLER_H


#include "dyros_jet_controller/dyros_jet_model.h"
#include "math_type_define.h"
#include <vector>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <mutex>

#define ZERO_LIBRARY_MODE


const int FILE_CNT = 15;

const std::string FILE_NAMES[FILE_CNT] =
{
  ///change this directory when you use this code on the other computer///

  "/home/myeongju/data/walking/0_desired_zmp_.txt",
  "/home/myeongju/data/walking/1_desired_com_.txt",
  "/home/myeongju/data/walking/2_desired_q_.txt",
  "/home/myeongju/data/walking/3_real_q_.txt",
  "/home/myeongju/data/walking/4_desired_swingfoot_.txt",
  "/home/myeongju/data/walking/5_desired_pelvis_trajectory_.txt",
  "/home/myeongju/data/walking/6_current_com_pelvis_trajectory_.txt",
  "/home/myeongju/data/walking/7_current_foot_trajectory_.txt",
  "/home/myeongju/data/walking/8_QPestimation_variables_.txt",
  "/home/myeongju/data/walking/9_ft_sensor_.txt",
  "/home/myeongju/data/walking/10_ext_encoder_.txt",
  "/home/myeongju/data/walking/11_kalman_estimator2_.txt",
  "/home/myeongju/data/walking/12_kalman_estimator1_.txt",
  "/home/myeongju/data/walking/13_kalman_estimator3_.txt",
  "/home/myeongju/data/walking/14_grav_torque_.txt"

};

using namespace std;
namespace dyros_jet_controller
{

class WalkingController
{
public:
  fstream file[FILE_CNT];


  static constexpr unsigned int PRIORITY = 8;


  WalkingController(DyrosJetModel& model, const VectorQd& current_q, const VectorQd& current_qdot, const double hz, const double& control_time) :
    total_dof_(DyrosJetModel::HW_TOTAL_DOF), model_(model), current_q_(current_q), current_qdot_(current_qdot), hz_(hz), current_time_(control_time), start_time_{}, end_time_{}, slowcalc_thread_(&WalkingController::slowCalc, this), calc_update_flag_(false), calc_start_flag_(false), ready_for_thread_flag_(false), ready_for_compute_flag_(false), foot_step_planner_mode_(false), walking_end_foot_side_ (false), foot_plan_walking_last_(false), foot_last_walking_end_(false)
  {
    walking_state_send = false;
    walking_end_ = false;
    /*for(int i=0; i<FILE_CNT;i++)
    {
      file[i].open(FILE_NAMES[i].c_str(),ios_base::out);
    }
    
    file[0]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"zmp_desired_(0)"<<"\t"<<"zmp_desired_(1)"<<"\t"<<"foot_step_(current_step_num_, 0)"<<"\t"<<"foot_step_(current_step_num_, 1)"<<"\t"<<"foot_step_support_frame_(current_step_num_, 0)"<<"\t"<<"foot_step_support_frame_(current_step_num_, 1)"<<"\t"<<"foot_step_support_frame_(current_step_num_, 2)"<<endl;
    file[1]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"com_desired_(0)"<<"\t"<<"com_desired_(1)"<<"\t"<<"com_desired_(2)"<<"\t"<<"com_dot_desired_(0)"<<"\t"<<"com_dot_desired_(1)"<<"\t"<<"com_dot_desired_(2)"<<"\t"<<"com_support_init_(0)"<<"\t"<<"com_support_init_(0)"<<"\t"<<"com_support_init_(0)"<<endl;
    file[2]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"desired_leg_q_(0)"<<"\t"<<"desired_leg_q_(1)"<<"\t"<<"desired_leg_q_(2)"<<"\t"<<"desired_leg_q_(3)"<<"\t"<<"desired_leg_q_(4)"<<"\t"<<"desired_leg_q_(5)"<<"\t"<<"desired_leg_q_(6)"<<"\t"<<"desired_leg_q_(7)"<<"\t"<<"desired_leg_q_(8)"<<"\t"<<"desired_leg_q_(9)"<<"\t"<<"desired_leg_q_(10)"<<"\t"<<"desired_leg_q_(11)"<<endl;
    file[3]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"current_q_(0)"<<"\t"<<"current_q_(1)"<<"\t"<<"current_q_(2)"<<"\t"<<"current_q_(3)"<<"\t"<<"current_q_(4)"<<"\t"<<"current_q_(5)"<<"\t"<<"current_q_(6)"<<"\t"<<"current_q_(7)"<<"\t"<<"current_q_(8)"<<"\t"<<"current_q_(9)"<<"\t"<<"current_q_(10)"<<"\t"<<"current_q_(11)"<<endl;
    file[4]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"rfoot_trajectory_support_.translation()(0)"<<"\t"<<"rfoot_trajectory_support_.translation()(1)"<<"\t"<<"rfoot_trajectory_support_.translation()(2)"<<"\t"<<"lfoot_trajectory_support_.translation()(0)"<<"\t"<<"lfoot_trajectory_support_.translation()(1)"<<"\t"<<"lfoot_trajectory_support_.translation()(2)"<<"\t"<<"rfoot_support_init_.translation()(0)"<<"\t"<<"rfoot_support_init_.translation()(1)"<<"\t"<<"rfoot_support_init_.translation()(2)"<<endl;
    file[5]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"pelv_trajectory_support_.translation()(0)"<<"\t"<<"pelv_trajectory_support_.translation()(1)"<<"\t"<<"pelv_trajectory_support_.translation()(2)"<<endl;
    file[6]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"com_support_current_(0)"<<"\t"<<"com_support_current_(1)"<<"\t"<<"com_support_current_(2)"
          <<"\t"<<"pelv_support_current_.translation()(0)"<<"\t"<<"pelv_support_current_.translation()(1)"<<"\t"<<"pelv_support_current_.translation()(2)"<<"\t"<<"com_support_dot_current_(0)"<<"\t"<<"com_support_dot_current_(1)"<<"\t"<<"com_support_dot_current_(2)"
         <<"\t"<<"com_sim_current_(0)"<<"\t"<<"com_sim_current_(1)"<<"\t"<<"com_sim_current_(2)"<<"\t"<<"com_sim_dot_current_(0)"<<"\t"<<"com_sim_dot_current_(1)"<<"\t"<<"com_sim_dot_current_(2)"<<endl;
    file[7]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"rfoot_support_current_.translation()(0)"<<"\t"<<"rfoot_support_current_.translation()(1)"<<"\t"<<"rfoot_support_current_.translation()(2)"
          <<"\t"<<"lfoot_support_current_.translation()(0)"<<"\t"<<"lfoot_support_current_.translation()(1)"<<"\t"<<"lfoot_support_current_.translation()(2)"<<endl;
    file[8]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"vars.x[0]"<<"\t"<<"vars.x[1]"<<"\t"<<"vars.x[2]"<<"\t"<<"vars.x[3]"<<"\t"<<"vars.x[4]"<<"\t"<<"vars.x[5]"<<"\t"<<"zmp_measured_(0)"<<"\t"<<"zmp_measured_(1)"<<"\t"<<"zmp_r_(0)"<<"\t"<<"zmp_r_(1)"<<"\t"<<"zmp_l_(0)"<<"\t"<<"zmp_l_(1)"<<endl;
    file[9]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"r_ft_(0)"<<"\t"<<"r_ft_(1)"<<"\t"<<"r_ft_(2)"<<"\t"<<"r_ft_(3)"<<"\t"<<"r_ft_(4)"<<"\t"<<"r_ft_(5)"<<"\t"<<"l_ft_(0)"<<"\t"<<"l_ft_(1)"<<"\t"<<"l_ft_(2)"<<"\t"<<"l_ft_(3)"<<"\t"<<"l_ft_(4)"<<"\t"<<"l_ft_(5)"<<endl;
    file[10]<<"walking_tick_"<<"\t"<<"current_step_num_"<<"\t"<<"current_link_q_leg_(0)"<<"\t"<<"current_link_q_leg_(1)"<<"\t"<<"current_link_q_leg_(2)"<<"\t"<<"current_link_q_leg_(3)"<<"\t"<<"current_link_q_leg_(4)"<<"\t"<<"current_link_q_leg_(5)"<<"\t"<<
              "current_link_q_leg_(6)"<<"\t"<<"current_link_q_leg_(7)"<<"\t"<<"current_link_q_leg_(8)"<<"\t"<<"current_link_q_leg_(9)"<<"\t"<<"current_link_q_leg_(10)"<<"\t"<<"current_link_q_leg_(11)"<<endl;
    file[11]<<"walking_tick_"<<"\t"<<"X_hat_post_2_(0)"<<"\t"<<"X_hat_post_2_(1)"<<"\t"<<"X_hat_post_2_(2)"<<"\t"<<"X_hat_post_2_(3)"<<"\t"<<"X_hat_post_2_(4)"<<"\t"<<"X_hat_post_2_(5)"<<"\t"<<"X_hat_post_2_(6)"<<"\t"<<"X_hat_post_2_(7)"<<endl;
    file[12]<<"walking_tick_"<<"\t"<<"X_hat_post_1_(0)"<<"\t"<<"X_hat_post_1_(1)"<<"\t"<<"X_hat_post_1_(2)"<<"\t"<<"X_hat_post_1_(3)"<<"\t"<<"X_hat_post_1_(4)"<<"\t"<<"X_hat_post_1_(5)"<<endl;
    file[13]<<"walking_tick_"<<"\t"<<"X_hat_post_3_(0)"<<"\t"<<"X_hat_post_3_(1)"<<"\t"<<"X_hat_post_3_(2)"<<"\t"<<"X_hat_post_3_(3)"<<"\t"<<"X_hat_post_3_(4)"<<"\t"<<"X_hat_post_3_(5)"<<endl;
    file[14]<<"walking_tick_"<<"\t"<<"grav_ground_torque_(0)"<<"\t"<<"grav_ground_torque_(1)"<<"\t"<<"grav_ground_torque_(2)"<<"\t"<<"grav_ground_torque_(3)"<<"\t"<<"grav_ground_torque_(4)"<<"\t"<<"grav_ground_torque_(5)"<<endl;
    */
  } 

  void compute();
  void setTarget(int walk_mode, bool hip_compensation, bool lqr, int ik_mode, bool heel_toe,
                 bool is_right_foot_swing, double x, double y, double z, double height, double theta,
                 double step_length, double step_length_y, bool walking_pattern);
  void setEnable(bool enable);
  void setFootPlan(int footnum, int startfoot, Eigen::MatrixXd footpose);
  void updateControlMask(unsigned int *mask);
  void writeDesired(const unsigned int *mask, VectorQd& desired_q);

  void parameterSetting();
  //functions in compute
  void getRobotState();
  void getComTrajectory_MJ();
  void getZmpTrajectory();
  void getPelvTrajectory();
  void getFootTrajectory_MJ();
  void computeIkControl_MJ(Eigen::Isometry3d float_trunk_transform, Eigen::Isometry3d float_lleg_transform, Eigen::Isometry3d float_rleg_transform, Eigen::Vector12d& desired_leg_q);
  void computeJacobianControl(Eigen::Isometry3d float_lleg_transform, Eigen::Isometry3d float_rleg_transform, Eigen::Vector3d float_lleg_transform_euler, Eigen::Vector3d float_rleg_transform_euler, Eigen::Vector12d& desired_leg_q_dot);
  void compensator();

  void supportToFloatPattern();
  void updateNextStepTime();
  void updateInitialState();
  void updateInitialState2();
  //functions for getFootStep()
  void floatToSupportFootstep();
  void addZmpOffset();
  void zmpGenerator(const unsigned int norm_size, const unsigned planning_step_num);
  void onestepZmp(unsigned int current_step_number, Eigen::VectorXd& temp_px, Eigen::VectorXd& temp_py);
  void modified_zmp_trajectory_update(Eigen::Vector3d& LFoot_desired,Eigen::Vector3d& RFoot_desired,Eigen::MatrixXd& Ref_ZMP, Eigen::MatrixXd& modified_Ref_ZMP);
  void modified_zmp_trajectory_update_MJ(int Preview_step, int tick, double Zmp_start_point_X, double Zmp_start_point_Y, Eigen::MatrixXd& modified_Ref_ZMP);
  void onestepZmp_MJ(unsigned int current_step_number, Eigen::VectorXd& temp_px, Eigen::VectorXd& temp_py);
  void OfflineCoM_MJ(unsigned int current_step_number, Eigen::VectorXd& temp_cx, Eigen::VectorXd& temp_cy);
  void calculateFootStepSeparate();
  void calculateFootStepTotal();
  void usingFootStepPlanner();

  //functions in compensator()
  void hipCompensator(); //reference Paper: http://dyros.snu.ac.kr/wp-content/uploads/2017/01/ICHR_2016_JS.pdf
  void hipCompensation();

  //PreviewController
  void modifiedPreviewControl_MJ();
  void previewControl(double dt, int NL, int tick, double x_i, double y_i, Eigen::Vector3d xs,
                      Eigen::Vector3d ys, double ux_1 , double uy_1 ,
                      double& ux, double& uy, double gi, Eigen::VectorXd gp_l,
                      Eigen::Matrix1x3d gx, Eigen::Matrix3d a, Eigen::Vector3d b,
                      Eigen::Matrix1x3d c, Eigen::Vector3d &xd, Eigen::Vector3d &yd);
  void previewControlParameter(double dt, int NL, Eigen::Matrix4d& k, Eigen::Vector3d com_support_init_,
                               double& gi, Eigen::VectorXd& gp_l, Eigen::Matrix1x3d& gx, Eigen::Matrix3d& a,
                               Eigen::Vector3d& b, Eigen::Matrix1x3d& c);
  
  void preview_MJ_CPM(double dt, int NL, int tick, double x_i, double y_i, Eigen::Vector3d xs, Eigen::Vector3d ys, double& UX, double& UY, 
       Eigen::MatrixXd Gi, Eigen::VectorXd Gd, Eigen::MatrixXd Gx, Eigen::MatrixXd A, Eigen::VectorXd B, Eigen::MatrixXd A_bar, Eigen::VectorXd B_bar, Eigen::Vector2d &XD, Eigen::Vector2d &YD, Eigen::VectorXd& X_bar_p, Eigen::VectorXd& Y_bar_p);
   
  void previewParam_MJ_CPM(double dt, int NL, Eigen::Matrix3d& K, Eigen::Vector3d com_support_init_, Eigen::MatrixXd& Gi, Eigen::VectorXd& Gd, Eigen::MatrixXd& Gx, 
  Eigen::MatrixXd& A, Eigen::VectorXd& B, Eigen::MatrixXd& C, Eigen::MatrixXd& D, Eigen::MatrixXd& A_bar, Eigen::VectorXd& B_bar);

  void preview_MJ(double dt, int NL, int tick, double x_i, double y_i, Eigen::Vector3d xs, Eigen::Vector3d ys, double& UX, double& UY, 
       Eigen::MatrixXd Gi, Eigen::VectorXd Gd, Eigen::MatrixXd Gx, Eigen::MatrixXd A, Eigen::VectorXd B, Eigen::Vector3d &XD, Eigen::Vector3d &YD);
  
  void previewParam_MJ(double dt, int NL, Eigen::Matrix4d& K, Eigen::Vector3d com_support_init_, Eigen::MatrixXd& Gi, Eigen::VectorXd& Gd, Eigen::MatrixXd& Gx, 
  Eigen::MatrixXd& A, Eigen::VectorXd& B, Eigen::MatrixXd& C, Eigen::MatrixXd& D, Eigen::MatrixXd& A_bar, Eigen::VectorXd& B_bar);


  void Ankle_ori_controller(int tick, double py_ref_);

  //LQR && External Encoder
  void vibrationControl(const Eigen::Vector12d desired_leg_q, Eigen::Vector12d &output);
  void vibrationControl_MJ(const Eigen::Vector12d desired_leg_q, Eigen::Vector12d &output);
  void massSpringMotorModel(double spring_k, double damping_d, double motor_k, Eigen::Matrix12d & mass, Eigen::Matrix<double, 36, 36>& a, Eigen::Matrix<double, 36, 12>& b, Eigen::Matrix<double, 12, 36>& c);
  void discreteModel(Eigen::Matrix<double, 36, 36>& a, Eigen::Matrix<double, 36, 12>& b, Eigen::Matrix<double, 12, 36>& c, int np, double dt,
                     Eigen::Matrix<double, 36, 36>& ad, Eigen::Matrix<double, 36, 12>& bd, Eigen::Matrix<double, 12, 36>& cd,
                     Eigen::Matrix<double, 48, 48>& ad_total, Eigen::Matrix<double, 48, 12>& bd_total);
  void riccatiGain(Eigen::Matrix<double, 48, 48>& ad_total, Eigen::Matrix<double, 48, 12>& bd_total, Eigen::Matrix<double, 48, 48>& q, Eigen::Matrix12d& r, Eigen::Matrix<double, 12, 48>& k);
  void slowCalc();
  void slowCalcContent();

  
  void discreteRiccatiEquationInitialize(Eigen::MatrixXd a, Eigen::MatrixXd b);
  Eigen::MatrixXd discreteRiccatiEquationLQR(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd R, Eigen::MatrixXd Q);
  Eigen::MatrixXd discreteRiccatiEquationPrev(Eigen::MatrixXd a, Eigen::MatrixXd b, Eigen::MatrixXd r, Eigen::MatrixXd q);

  VectorQd desired_q_not_compensated_;

  bool walking_end_foot_side_;
  bool walking_end_;
  bool foot_plan_walking_last_;
  bool foot_last_walking_end_;
  bool walking_state_send;


  //ImpedanceControl
  void impedancefootUpdate();
  void impedanceControl();

  //CapturePoint
  void getCapturePointTrajectory();
  void getCapturePoint_init_ref();
  void CapturePointModify();
  void zmptoInitFloat();
  Eigen::VectorXd capturePoint_refx, capturePoint_refy;
  Eigen::VectorXd zmp_refx, zmp_refy;
  Eigen::VectorXd capturePoint_ox, capturePoint_oy, zmp_dx, zmp_dy;
  Eigen::Vector3d capturePoint_measured_;
  double last_time_;
  int capturePoint_current_num_;
  Eigen::Vector3d com_float_prev_;
  Eigen::Vector4d com_float_prev_dot_;
  Eigen::Vector4d com_float_prev;
  Eigen::Vector3d com_support_prev;
  double ux_1, uy_1;
  Eigen::Vector3d xs, ys;
  int currentstep;
  bool firsttime = false;
  Eigen::Vector2d capturePoint_offset_;
  Eigen::Isometry3d float_support_init;
  Eigen::Isometry3d current_step_float_support_;

  Eigen::Isometry3d support_float_init;
  Eigen::Isometry3d current_step_support_float_;

  Eigen::Vector6d q_sim_virtual_;
  Eigen::Vector6d q_sim_dot_virtual_;
  Eigen::VectorXd com_refx;
  Eigen::VectorXd com_refy;
  Eigen::VectorXd com_dot_refx;
  Eigen::VectorXd com_dot_refy;
  Eigen::Vector2d com_initx;
  Eigen::Vector2d com_inity;


private:

  const double hz_;
  const double &current_time_; // updated by control_base
  unsigned int walking_tick_ = 0;
  unsigned int com_tick_ = 0;
  double walking_time_ = 0;
  unsigned int tick_d1 = 0, tick_d2 = 0;
  //sensorData
  Eigen::Vector6d r_ft_;
  Eigen::Vector6d l_ft_;
  Eigen::Vector3d imu_acc_;
  Eigen::Vector3d imu_ang_;
  Eigen::Vector3d imu_grav_rpy_;

  double total_mass = 0;

  //parameterSetting()
  double t_last_;
  double t_start_;
  double t_start_real_;
  double t_temp_;
  double t_imp_;
  double t_rest_init_;
  double t_rest_last_;
  double t_double1_;
  double t_double2_;
  double t_total_;
  double foot_height_;
  double com_height_;

  bool com_control_mode_;
  bool com_update_flag_; // frome A to B
  bool gyro_frame_flag_;
  bool ready_for_thread_flag_;
  bool ready_for_compute_flag_;
  bool estimator_flag_;

  int ik_mode_;
  int walk_mode_;
  bool hip_compensator_mode_;
  bool lqr_compensator_mode_;
  int heel_toe_mode_;
  int is_right_foot_swing_;
  bool foot_step_planner_mode_;

  bool walking_enable_;
  bool joint_enable_[DyrosJetModel::HW_TOTAL_DOF];
  double step_length_x_;
  double step_length_y_;

  //double step_angle_theta_;
  unsigned int print_flag = 0;
  unsigned int print_flag_1 = 0;
  double target_x_;
  double target_y_;
  double target_z_;
  double target_theta_;
  double total_step_num_;
  double current_step_num_;
  int foot_step_plan_num_;
  int foot_step_start_foot_;
  bool walkingPatternDCM_;
  Eigen::MatrixXd foot_pose_;

  Eigen::MatrixXd foot_step_;
  Eigen::MatrixXd foot_step_support_frame_;
  Eigen::MatrixXd foot_step_support_frame_offset_;

  Eigen::MatrixXd org_ref_zmp_;
  Eigen::MatrixXd ref_zmp_;
  Eigen::MatrixXd modified_ref_zmp_;
  Eigen::MatrixXd ref_com_;
  Eigen::MatrixXd ref_zmp_float_;

  VectorQd start_q_;
  VectorQd desired_q_;
  VectorQd target_q_;
  const VectorQd& current_q_;
  const VectorQd& current_qdot_;

  double prev_zmp_error_y = 0, prev_zmp_error_x = 0;


  //const double &current_time_;
  const unsigned int total_dof_;
  double start_time_[DyrosJetModel::HW_TOTAL_DOF];
  double end_time_[DyrosJetModel::HW_TOTAL_DOF];

  //Step initial state variable//
  Eigen::Isometry3d pelv_support_init_2;
  Eigen::Isometry3d pelv_support_init_;
  Eigen::Isometry3d lfoot_support_init_;
  Eigen::Isometry3d rfoot_support_init_;
  Eigen::Isometry3d pelv_float_init_;
  Eigen::Isometry3d lfoot_float_init_;
  Eigen::Isometry3d rfoot_float_init_;

  Eigen::Vector3d pelv_support_euler_init_;
  Eigen::Vector3d lfoot_support_euler_init_;
  Eigen::Vector3d rfoot_support_euler_init_;
  VectorQd q_init_;

  Eigen::Vector6d supportfoot_float_init_;
  Eigen::Vector6d supportfoot_support_init_;
  Eigen::Vector6d supportfoot_support_init_offset_;
  Eigen::Vector6d swingfoot_float_init_;
  Eigen::Vector6d swingfoot_support_init_;
  Eigen::Vector6d swingfoot_support_init_offset_;

  Eigen::Isometry3d pelv_support_start_;

  Eigen::Vector3d com_float_init_;
  Eigen::Vector3d com_support_init_;
  Eigen::Vector3d com_support_init_2;
  double lfoot_zmp_offset_;   //have to be initialized
  double rfoot_zmp_offset_;
  Eigen::Vector3d com_offset_;

  //Step current state variable//
  Eigen::Vector3d com_support_current_CLIPM_Euler;
  Eigen::Vector3d com_support_current_CLIPM_b;
  Eigen::Vector3d com_support_current_CLIPM;
  Eigen::Vector3d com_support_current_b;
  Eigen::Vector3d com_support_current_dot;
  Eigen::Vector3d com_support_current_;
  Eigen::Vector3d com_support_current_Euler;
  Eigen::Vector3d com_middle_support_current_;
  Eigen::Vector3d com_support_dot_current_;//from support foot
  Eigen::Vector3d com_support_ddot_current_;//from support foot 

  ///simulation
  Eigen::Vector3d com_sim_current_;
  Eigen::Vector3d com_sim_dot_current_;
  Eigen::Isometry3d lfoot_sim_global_current_;
  Eigen::Isometry3d rfoot_sim_global_current_;
  Eigen::Isometry3d base_sim_global_current_;
  Eigen::Isometry3d lfoot_sim_float_current_;
  Eigen::Isometry3d rfoot_sim_float_current_;
  Eigen::Isometry3d supportfoot_float_sim_current_;

  Eigen::Vector3d gyro_sim_current_;
  Eigen::Vector3d accel_sim_current_;
  
  Eigen::Isometry3d supportfoot_float_current_Euler;
  Eigen::Isometry3d supportfoot_float_current_;
  Eigen::Isometry3d pelv_support_current_Euler;
  Eigen::Isometry3d pelv_support_current_;
  Eigen::Isometry3d lfoot_support_current_;
  Eigen::Isometry3d rfoot_support_current_;
  Eigen::Isometry3d lfoot_support_current_ZMP;
  Eigen::Isometry3d rfoot_support_current_ZMP;

  Eigen::Vector3d com_float_current_;
  Eigen::Vector3d com_float_current_RPY;
  Eigen::Vector3d com_float_current_Euler;
  Eigen::Vector3d com_float_current_dot_;
  Eigen::Isometry3d pelv_float_current_;
  Eigen::Isometry3d lfoot_float_current_;
  Eigen::Isometry3d rfoot_float_current_;
  Eigen::Isometry3d lfoot_float_current_Euler;
  Eigen::Isometry3d rfoot_float_current_Euler;
  Eigen::Isometry3d R_;
  Eigen::Matrix3d R;
  
  Eigen::Matrix6d current_leg_jacobian_l_;
  Eigen::Matrix6d current_leg_jacobian_r_;
  DyrosJetModel &model_;

  Eigen::Vector3d com_desired_float_;
  double final_ref_zmp_print = 0 ;
  double final_com_print = 0 ;
  //desired variables
  Eigen::Vector12d q_des;
  Eigen::Vector12d desired_leg_q_;
  Eigen::Vector12d desired_leg_q_dot_;
  Eigen::Vector3d com_desired_;
  Eigen::Vector3d com_dot_desired_;
  Eigen::Vector2d zmp_desired_;
  // 수업용
  Eigen::Vector3d com_desired_dot_, com_desired_ddot_ , com_desired_dot_b_;
  //
  Eigen::Isometry3d rfoot_trajectory_support_;  //local frame
  Eigen::Isometry3d lfoot_trajectory_support_;
  Eigen::Vector3d rfoot_trajectory_euler_support_;
  Eigen::Vector3d lfoot_trajectory_euler_support_;
  Eigen::Vector6d rfoot_trajectory_dot_support_; //x,y,z translation velocity & roll, pitch, yaw velocity
  Eigen::Vector6d lfoot_trajectory_dot_support_;
 
  Eigen::Isometry3d pelv_trajectory_support_; //local frame
  Eigen::Isometry3d pelv_trajectory_float_; //pelvis frame
 //
  Eigen::Isometry3d rfoot_trajectory_float_;  //pelvis frame
  Eigen::Isometry3d lfoot_trajectory_float_;
  Eigen::Vector3d rfoot_trajectory_euler_float_;
  Eigen::Vector3d lfoot_trajectory_euler_float_;
  Eigen::Vector3d rfoot_trajectory_dot_float_;
  Eigen::Vector3d lfoot_trajectory_dot_float_;

  //getComTrajectory() variables
  double xi_;
  double yi_;
  Eigen::Vector3d xs_;
  Eigen::Vector3d ys_;
  Eigen::Vector3d xd_;
  Eigen::Vector3d yd_; 

  //Preview Control
  Eigen::Vector3d preview_x, preview_y, preview_x_b, preview_y_b, preview_x_b2, preview_y_b2;
  double ux_, uy_, ux_1_, uy_1_;
  double zc_;
  double gi_;
  double zmp_start_time_; //원래 코드에서는 start_time, zmp_ref 시작되는 time같음
  Eigen::Matrix4d k_;
  Eigen::Matrix4d K_act_;
  Eigen::VectorXd gp_l_;
  Eigen::Matrix1x3d gx_;
  Eigen::Matrix3d a_;
  Eigen::Vector3d b_;
  Eigen::Matrix1x3d c_;

  //Preview CLIPM MJ
  Eigen::MatrixXd A_;
  Eigen::VectorXd B_;
  Eigen::MatrixXd C_;
  Eigen::MatrixXd D_;
  Eigen::Matrix3d K_;
  Eigen::MatrixXd Gi_;
  Eigen::MatrixXd Gx_;
  Eigen::VectorXd Gd_;
  Eigen::MatrixXd A_bar_;
  Eigen::VectorXd B_bar_;
  Eigen::Vector2d Preview_X, Preview_Y, Preview_X_b, Preview_Y_b;
  Eigen::VectorXd X_bar_p_, Y_bar_p_;
  Eigen::Vector2d XD_;
  Eigen::Vector2d YD_;
  double UX_, UY_; 

  //CP Feedback MJ
  double R_angle = 0, P_angle = 0;
  double cp_err_x = 0, cp_err_y = 0; 
  double cp_err_x_i = 0, cp_err_y_i = 0;
  double del_px_cp = 0, del_py_cp = 0; 

  double x_cp_ref = 0, y_cp_ref = 0; 
  double x_cp_act = 0, y_cp_act = 0; 
  double Wn;  
  
  // 속도 구할때
  double XD_b = 0, YD_b = 0, XD_bb = 0, YD_bb = 0;
  double X_b = 0, Y_b = 0, X_bb = 0, Y_bb = 0;
  double XD_vel = 0, YD_vel = 0, XD_vel_b = 0, YD_vel_b = 0;
  double XA_vel = 0, YA_vel = 0, XA_vel_b = 0, YA_vel_b = 0;
  // ZMP Controller_MJ

  double px_act = 0, py_act = 0; 
  double px_err = 0, x_ddot_des = 0, x_dot_des = 0, x_des = 0, x_des_support = 0;
  double py_err = 0, y_ddot_des = 0, y_dot_des = 0, y_des = 0, y_des_support = 0;
  // 

  //resolved momentum control
  Eigen::Vector3d p_ref_;
  Eigen::Vector3d l_ref_;

  //Gravitycompensate
  Eigen::Vector12d joint_offset_angle_;
  Eigen::Vector12d grav_ground_torque_;

  //vibrationCotrol
  std::mutex slowcalc_mutex_;
  std::thread slowcalc_thread_;

  Eigen::Vector12d current_motor_q_leg_;
  Eigen::Vector12d current_link_q_leg_;
  Eigen::Vector12d pre_motor_q_leg_;
  Eigen::Vector12d pre_link_q_leg_;
  Eigen::Vector12d lqr_output_;
  Eigen::Vector12d lqr_output_pre_;
  Eigen::Vector12d DOB_IK_output_;
  Eigen::Vector12d DOB_IK_output_b_;

  VectorQd thread_q_;
  unsigned int thread_tick_;

  Eigen::Matrix<double, 48, 1> x_bar_right_;
  Eigen::Matrix<double, 12, 48> kkk_copy_;
  Eigen::Matrix<double, 48, 48> ad_total_copy_;
  Eigen::Matrix<double, 48, 12> bd_total_copy_;
  Eigen::Matrix<double, 36, 36> ad_copy_;
  Eigen::Matrix<double, 36, 12> bd_copy_;

  Eigen::Matrix<double, 36, 36> ad_right_;

  Eigen::Matrix<double, 36, 12> bd_right_;
  Eigen::Matrix<double, 48, 48> ad_total_right_;
  Eigen::Matrix<double, 48, 12> bd_total_right_;
  Eigen::Matrix<double, 12, 48> kkk_motor_right_;

  Eigen::Vector12d d_hat_b;

  bool calc_update_flag_;
  bool calc_start_flag_;


  Eigen::Matrix<double, 18, 18> mass_matrix_;
  Eigen::Matrix<double, 18, 18> mass_matrix_pc_;
  Eigen::Matrix<double, 12, 12> mass_matrix_sel_;
  Eigen::Matrix<double, 36, 36> a_right_mat_;
  Eigen::Matrix<double, 36, 12> b_right_mat_;
  Eigen::Matrix<double, 12, 36> c_right_mat_;
  Eigen::Matrix<double, 36, 36> a_disc_;
  Eigen::Matrix<double, 36, 12> b_disc_;
  Eigen::Matrix<double, 48, 48> a_disc_total_;
  Eigen::Matrix<double, 48, 12> b_disc_total_;
  Eigen::Matrix<double, 48, 48> kkk_;


  //////////////////QP based StateEstimation/////////////////////
  Eigen::Matrix<double, 18, 6> a_total_;
  Eigen::Matrix<double, 2, 6> a_kin_;
  Eigen::Matrix<double, 2, 6> a_c_dot_;
  Eigen::Matrix<double, 2, 6> a_c_;
  Eigen::Matrix<double, 2, 6> a_zmp_;
  Eigen::Matrix<double, 2, 6> a_c_c_dot_;
  Eigen::Matrix<double, 2, 6> a_f_;
  Eigen::Matrix<double, 6, 6> a_noise_;
  Eigen::Matrix<double, 18, 1> b_total_;
  Eigen::Matrix<double, 2, 1> b_kin_;
  Eigen::Matrix<double, 2, 1> b_c_dot_;
  Eigen::Matrix<double, 2, 1> b_c_;
  Eigen::Matrix<double, 2, 1> b_zmp_;
  Eigen::Matrix<double, 2, 1> b_c_c_dot_;
  Eigen::Matrix<double, 2, 1> b_f_;
  Eigen::Matrix<double, 6, 1> b_noise_;


  Eigen::Vector3d com_float_old_;
  Eigen::Vector3d com_float_dot_old_;
  Eigen::Vector3d com_support_old_;
  Eigen::Vector3d com_support_dot_old_;
  Eigen::Vector3d com_sim_old_;
  Eigen::Vector2d com_support_dot_old_estimation_;
  Eigen::Vector2d com_support_old_estimation_;


  Eigen::Vector2d zmp_r_;
  Eigen::Vector2d zmp_l_;
  Eigen::Vector2d zmp_measured_;
  Eigen::Vector2d zmp_old_estimation_;

  Eigen::Vector6d x_estimation_;


  //Riccati variable
  Eigen::MatrixXd Z11;
  Eigen::MatrixXd Z12;
  Eigen::MatrixXd Z21;
  Eigen::MatrixXd Z22;
  Eigen::MatrixXd temp1;
  Eigen::MatrixXd temp2;
  Eigen::MatrixXd temp3;
  std::vector<double> eigVal_real; //eigen valueÀÇ real°ª
  std::vector<double> eigVal_img; //eigen valueÀÇ img°ª
  std::vector<Eigen::VectorXd> eigVec_real; //eigen vectorÀÇ real°ª
  std::vector<Eigen::VectorXd> eigVec_img; //eigen vectorÀÇ img°ª
  Eigen::MatrixXd Z;
  Eigen::VectorXd deigVal_real;
  Eigen::VectorXd deigVal_img;
  Eigen::MatrixXd deigVec_real;
  Eigen::MatrixXd deigVec_img;
  Eigen::MatrixXd tempZ_real;
  Eigen::MatrixXd tempZ_img;
  Eigen::MatrixXcd U11_inv;
  Eigen::MatrixXcd X;
  Eigen::MatrixXd X_sol;

  Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType Z_eig;
  Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType es_eig;
  Eigen::MatrixXcd tempZ_comp;
  Eigen::MatrixXcd U11;
  Eigen::MatrixXcd U21;

  void getQpEstimationInputMatrix();
  ////////////////////////////////////////////////////////


  /////////////////////////Kalman Filter1///////////////////////
  Eigen::Matrix<double, 6, 6> Ad_1_;
  Eigen::Matrix<double, 6, 2> Bd_1_;
  Eigen::Matrix<double, 4, 6> Cd_1_;
  Eigen::Matrix<double, 6, 6> Q_1_;
  Eigen::Matrix<double, 4, 4> R_1_;


  Eigen::Matrix<double, 6, 1> X_hat_prio_1_;
  Eigen::Matrix<double, 6, 1> X_hat_post_1_;
  Eigen::Matrix<double, 6, 1> X_hat_prio_old_1_;
  Eigen::Matrix<double, 6, 1> X_hat_post_old_1_;

  Eigen::Matrix<double, 4, 1> Y_1_;



  Eigen::Matrix<double, 6, 6> P_prio_1_;
  Eigen::Matrix<double, 6, 6> P_post_1_;
  Eigen::Matrix<double, 6, 6> P_prio_old_1_;
  Eigen::Matrix<double, 6, 6> P_post_old_1_;

  Eigen::Matrix<double, 6, 4> K_1_;
  Eigen::Matrix<double, 6, 4> K_old_1_;


  Eigen::Matrix<double, 2, 1> u_old_1_;

  void kalmanFilter1();
  void kalmanStateSpace1();
  //////////////////////////////////////////////////////////////


  /////////////////////////Kalman Filter2///////////////////////

  Eigen::Matrix<double, 8, 8> Ad_2_;
  Eigen::Matrix<double, 8, 2> Bd_2_;
  Eigen::Matrix<double, 4, 8> Cd_2_;
  Eigen::Matrix<double, 8, 8> Q_2_;
  Eigen::Matrix<double, 4, 4> R_2_;


  Eigen::Matrix<double, 8, 1> X_hat_prio_2_;
  Eigen::Matrix<double, 8, 1> X_hat_post_2_;
  Eigen::Matrix<double, 8, 1> X_hat_prio_old_2_;
  Eigen::Matrix<double, 8, 1> X_hat_post_old_2_;

  Eigen::Matrix<double, 4, 1> Y_2_;



  Eigen::Matrix<double, 8, 8> P_prio_2_;
  Eigen::Matrix<double, 8, 8> P_post_2_;
  Eigen::Matrix<double, 8, 8> P_prio_old_2_;
  Eigen::Matrix<double, 8, 8> P_post_old_2_;

  Eigen::Matrix<double, 8, 4> K_2_;
  Eigen::Matrix<double, 8, 4> K_old_2_;

  Eigen::Matrix<double, 2, 1> u_old_2_;


  void kalmanFilter2();
  void kalmanStateSpace2();
  //////////////////////////////////////////////////////////////


  /////////////////////////Kalman Filter3///////////////////////

  Eigen::Matrix<double, 10, 10> Ad_3_;
  Eigen::Matrix<double, 10, 2> Bd_3_;
  Eigen::Matrix<double, 6, 10> Cd_3_;
  Eigen::Matrix<double, 10, 10> Q_3_;
  Eigen::Matrix<double, 6, 6> R_3_;


  Eigen::Matrix<double, 10, 1> X_hat_prio_3_;
  Eigen::Matrix<double, 10, 1> X_hat_post_3_;
  Eigen::Matrix<double, 10, 1> X_hat_prio_old_3_;
  Eigen::Matrix<double, 10, 1> X_hat_post_old_3_;

  Eigen::Matrix<double, 6, 1> Y_3_;



  Eigen::Matrix<double, 10, 10> P_prio_3_;
  Eigen::Matrix<double, 10, 10> P_post_3_;
  Eigen::Matrix<double, 10, 10> P_prio_old_3_;
  Eigen::Matrix<double, 10, 10> P_post_old_3_;

  Eigen::Matrix<double, 10, 6> K_3_;
  Eigen::Matrix<double, 10, 6> K_old_3_;


  Eigen::Matrix<double, 2, 1> u_old_3_;

  void kalmanFilter3();
  void kalmanStateSpace3();
  //////////////////////////////////////////////////////////////

};

}
#endif // WALKING_CONTROLLER_H