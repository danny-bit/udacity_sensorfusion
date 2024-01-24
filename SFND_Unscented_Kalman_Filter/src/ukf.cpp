#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)  MatrixXd Q_aug_ = MatrixXd(2,2);
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  n_x_= 5;
  n_v_ = 2;
  n_aug_ = n_x_+n_v_;
  n_sig_ = 2*n_aug_+1;
  lambda_ = 3-n_aug_;

  // init: sigma point weights
  weights_ = VectorXd(n_sig_);
  weights_.fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  //init: state-prediction
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_pred_.fill(0.0);

  //init: measurement co-variances
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;

  // init: augmentation covariance matrix
  Q_aug_ = MatrixXd(2,2);
  Q_aug_.setZero();
  Q_aug_.diagonal() << std_a_*std_a_, std_yawdd_*std_yawdd_;

  frame_cnt_ = 0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*
    measurement_package received -> process
    switch between sensors (lidar and radar)
   */
  if ( !is_initialized_) {
    // initialize state and covariace matrix
    x_.fill(0.0);
    P_.setIdentity();
    P_(3,3)=10.0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // RADAR: initialize state vector
      double rho = meas_package.raw_measurements_[0];     // range (dist from origin)
      double phi = meas_package.raw_measurements_[1];     // bearing (angle to x axis)
      double rho_dot = meas_package.raw_measurements_[2]; // velocity in radial direction
      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double vx = rho_dot * cos(phi);
  	  double vy = rho_dot * sin(phi);
      double v = sqrt(vx * vx + vy * vy);

      x_.head(3) << x, y, rho_dot;
      //P_(3,3)=std_radr_*std_radr_;
    } 
    else 
    {
      // LIDAR: initialize state vector
      x_.head(2) << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
      //P_(1, 1) = std_laspx_*std_laspx_;
      //P_(2, 2) = std_laspy_*std_laspy_;
    }


    time_us_ = meas_package.timestamp_ ;

    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_);
  delta_t /= 1000000.0;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }

  time_us_ = meas_package.timestamp_;
  frame_cnt_ = frame_cnt_ + 1;
}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  //### AUGMENTATION
  // augmented state vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_)      = 0;
  x_aug(n_x_+1)    = 0;

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_); P_aug.setZero();
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug.bottomRightCorner(n_v_,n_v_) = Q_aug_;
 
  // create X_aug
  MatrixXd L = P_aug.llt().matrixL();

  MatrixXd Xsig_aug = MatrixXd(n_aug_,n_sig_); Xsig_aug.setZero();

  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  if (frame_cnt_ == 1)
  {
  std::cout << x_ << std::endl;
  std::cout << P_aug << std::endl;
  std::cout << Xsig_aug << std::endl;
  }
  
  //### PREDICT SIGMA POINTS
  Xsig_pred_.fill(0.0);
  for (int i = 0; i< n_sig_; ++i) {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
  //### PREDICT (convert sigma point to prediction)
  x_.fill(0.0);
  x_ = Xsig_pred_ * weights_;

  P_.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) { 
    VectorXd x_diff = Xsig_pred_.col(i) - x_; // state difference
    x_diff(3) = remainder (x_diff(3),M_PI); // normalize angle

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // ###### PREDICT SIGMA POINTS LIDAR MODEL
  int n_z = 2;
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred = Zsig*weights_;

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z); S.fill(0.0);

  for (int i = 0; i < n_sig_; i++) { 
    VectorXd z_diff = Zsig.col(i) - z_pred;     //residual
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_lidar_;

  
  VectorXd z = meas_package.raw_measurements_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z); Tc.fill(0.0);

  for (int i = 0; i < n_sig_; ++i) { 
    VectorXd z_diff = Zsig.col(i) - z_pred;// residual     
    VectorXd x_diff = Xsig_pred_.col(i) - x_; // state difference
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // ###### UKF UPDATE
  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z,n_sig_);
  Zsig.fill(0.0);

   // transform sigma points into measurement space (2n+1 sigma points)
  for (int i = 0; i < n_sig_; ++i) {  
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    p_x = (fabs(p_x)<0.001) ? 0.001 : p_x;
    p_y = (fabs(p_y)<0.001) ? 0.001 : p_y;

    // measurement model
    double r = sqrt(p_x*p_x + p_y*p_y);
    Zsig(0,i) = r;
    Zsig(1,i) = atan2(p_y,p_x);
    Zsig(2,i) = (p_x*v*cos(yaw) + p_y*v*sin(yaw)) / r;
  }

  // mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  // innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z); S.setZero();
  
  for (int i = 0; i < n_sig_; ++i) { 
    
    VectorXd z_diff = Zsig.col(i) - z_pred; // residual
    z_diff(1) = remainder (z_diff(1),M_PI); // normalize angle
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R_radar_; // add measurement noise covariance matrix

  VectorXd z = meas_package.raw_measurements_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z); Tc.setZero();
  
  for (int i = 0; i < n_sig_; ++i) {  
    VectorXd z_diff = Zsig.col(i) - z_pred; // sigma point residual
    VectorXd x_diff = Xsig_pred_.col(i) - x_; // state difference
    z_diff(1) = remainder (z_diff(1),M_PI); // normalize angle
    x_diff(3) = remainder (x_diff(3),M_PI); // normalize angle
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // # UPDATE
  MatrixXd K = Tc * S.inverse(); // Kalman gain K;
  VectorXd z_diff = z - z_pred; // residual
  z_diff(1) = remainder (z_diff(1),M_PI); // normalize angle
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}