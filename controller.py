import rospy
import math
import numpy as np
from std_msgs.msg import Bool
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, VehicleSpeedRpt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import time

###############################################################################
# Lane Following Controller
#
# This module implements control algorithms for autonomous vehicle steering
# and speed regulation based on detected lane markers. It uses PID controllers
# for both lateral (steering) and longitudinal (speed) control.
###############################################################################

class PID:
    """
    Generic PID controller implementation with anti-windup protection.
    
    This class provides a flexible PID controller that can be used for
    various control applications including steering and speed control.
    
    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        wg (float): Windup guard limit (anti-windup)
    """
    def __init__(self, kp, ki, kd, wg=None):
        """
        Initialize PID controller with gains and optional windup guard.
        
        Args:
            kp: Proportional gain - immediate response to error
            ki: Integral gain - compensates for steady-state error
            kd: Derivative gain - provides damping effect
            wg: Windup guard limit - prevents integral term from growing too large
        """
        self.iterm = 0            # Integral term accumulator
        self.last_t = None        # Time of last control computation
        self.last_e = 0           # Previous error value
        self.kp = kp              # Proportional gain
        self.ki = ki              # Integral gain
        self.kd = kd              # Derivative gain
        self.wg = wg              # Windup guard limit
        self.derror = 0           # Derivative of error (rate of change)

    def reset(self):
        """
        Reset controller state.
        
        Call this function when restarting control or changing setpoints
        significantly to prevent accumulated integral term from causing
        undesired behavior.
        """
        self.iterm = 0        # Clear integral accumulator
        self.last_e = 0       # Reset previous error
        self.last_t = None    # Reset timing information

    def get_control(self, t, e, fwd=0):
        """
        Calculate control output based on error and time.
        
        Implements standard PID control algorithm:
        u(t) = Kp*e(t) + Ki∫e(τ)dτ + Kd*de/dt + feedforward
        
        Args:
            t: Current time in seconds
            e: Current error value (setpoint - measured value)
            fwd: Optional feedforward term
            
        Returns:
            Control output value
        """
        # Handle first iteration (no derivative)
        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            # Calculate error derivative (rate of change)
            de = (e - self.last_e) / (t - self.last_t)

        # Outlier detection - if error jumps too much, ignore derivative term
        # This prevents large control actions due to sensor noise or outliers
        if abs(e - self.last_e) > 0.5:
            de = 0

        # Update integral term (∫e(τ)dτ)
        self.iterm += e * (t - self.last_t)

        # Apply anti-windup guard to limit integral term
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg        # Cap positive integral term
            elif self.iterm < -self.wg:
                self.iterm = -self.wg       # Cap negative integral term

        # Store current values for next iteration
        self.last_e = e
        self.last_t = t
        self.derror = de

        # Return PID control output + feedforward term
        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class LaneFollowController:
    """
    Controller for autonomous lane following.
    
    This class implements a closed-loop control system that:
    1. Takes lane detection information as input
    2. Controls vehicle steering to follow detected lane
    3. Regulates vehicle speed based on desired setpoint
    4. Manages vehicle actuation through PACMOD interface
    
    The controller uses separate PID controllers for steering and speed control.
    """
    
    def __init__(self):
        """
        Initialize lane following controller with parameters and ROS connections.
        
        Sets up:
        - PID controllers for steering and speed
        - ROS subscribers for sensor inputs
        - ROS publishers for vehicle actuators
        - Vehicle state variables and command messages
        """
        # Initialize ROS node
        rospy.init_node('lane_follow_controller', anonymous=True)
        self.rate = rospy.Rate(10)  # 10 Hz control loop

        ###############################################################################
        # Controller Parameters
        ###############################################################################
        
        # Vehicle control parameters
        self.desired_speed = 1       # Target speed in m/s
        self.max_accel = 2.5         # Maximum acceleration command (0-1 scale)
        self.image_width = 1280      # Camera image width in pixels
        self.image_center_x = self.image_width / 2.0  # Image center x-coordinate
        
        # Initialize PID controllers
        self.pid_speed = PID(kp=0.5, ki=0.0, kd=0.1, wg=20)  # Speed controller with windup guard
        self.pid_steer = PID(kp=0.01, ki=0.0, kd=0.005)      # Steering controller

        ###############################################################################
        # Vehicle State Variables
        ###############################################################################
        
        # Current vehicle state
        self.speed = 0.0             # Current speed in m/s
        self.endgoal_x = None        # Target x-coordinate in image plane (from lane detection)
        self.endgoal_y = None        # Target y-coordinate in image plane
        
        # System state flags
        self.gem_enable = False      # Flag indicating if GEM vehicle is enabled
        self.pacmod_enable = False   # Flag indicating if PACMOD interface is enabled

        ###############################################################################
        # ROS Subscribers
        ###############################################################################
        
        # Subscribe to vehicle status topics
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)
        self.speed_sub = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        
        # Subscribe to lane detection output
        self.endgoal_sub = rospy.Subscriber("/lane_detection/endgoal", PoseStamped, self.endgoal_callback)
        
        ###############################################################################
        # ROS Publishers
        ###############################################################################
        
        # Publishers for vehicle control commands
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        ###############################################################################
        # Command Messages Setup
        ###############################################################################
        
        # Enable command (used to enable autonomous mode)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False
        
        # Gear command (PARK = 0, NEUTRAL = 1, REVERSE = 2, FORWARD = 3)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 3  # Set to FORWARD gear
        
        # Brake command
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear = True
        self.brake_cmd.ignore = True
        
        # Acceleration command
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear = True
        self.accel_cmd.ignore = True
        
        # Turn signal command (LEFT = 0, NONE = 1, RIGHT = 2)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1  # No turn signal
        
        # Steering command
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0  # Steering angle in radians
        self.steer_cmd.angular_velocity_limit = 3.5  # Maximum steering rate

    def enable_callback(self, msg):
        """
        Callback for PACMOD enable status messages.
        
        Updates controller state when PACMOD autonomous mode is enabled/disabled.
        
        Args:
            msg: Bool message indicating PACMOD enable status
        """
        self.pacmod_enable = msg.data

    def speed_callback(self, msg):
        """
        Callback for vehicle speed report messages.
        
        Updates current vehicle speed for use in speed control loop.
        
        Args:
            msg: VehicleSpeedRpt message containing current speed
        """
        self.speed = round(msg.vehicle_speed, 3)

    def endgoal_callback(self, msg):
        """
        Callback for lane detection endgoal messages.
        
        Updates target position for steering control based on lane detection.
        
        Args:
            msg: PoseStamped message containing target position in image
        """
        self.endgoal_x = msg.pose.position.x
        self.endgoal_y = msg.pose.position.y

    def front2steer(self, f_angle):
        """
        Convert front wheel angle to steering wheel angle.
        
        This function implements the non-linear mapping between desired front
        wheel angle and required steering wheel angle based on vehicle geometry.
        
        Args:
            f_angle: Desired front wheel angle in degrees
            
        Returns:
            Required steering wheel angle in degrees
        """
        # Safety limits for front wheel angle
        if f_angle > 35:
            f_angle = 35  # Maximum right turn
        if f_angle < -35:
            f_angle = -35  # Maximum left turn
            
        # Non-linear mapping based on vehicle-specific calibration
        if f_angle > 0:
            # Right turn mapping
            steer_angle = round(-0.1084 * f_angle**2 + 21.775 * f_angle, 2)
        elif f_angle < 0:
            # Left turn mapping (use same curve but negate result)
            f_angle_p = -f_angle
            steer_angle = -round(-0.1084 * f_angle_p**2 + 21.775 * f_angle_p, 2)
        else:
            # No steering
            steer_angle = 0.0
            
        return steer_angle

    def start_control(self):
        """
        Main control loop for autonomous lane following.
        
        This function runs continuously and:
        1. Checks if vehicle is enabled and initializes it if needed
        2. Updates steering based on lane position feedback
        3. Controls speed through acceleration commands
        4. Publishes all control commands to PACMOD interface
        
        The loop runs at the rate specified by self.rate (10 Hz)
        """
        while not rospy.is_shutdown():
            ###############################################################################
            # Vehicle Initialization
            ###############################################################################
            
            # Enable vehicle if PACMOD is ready but vehicle not yet enabled
            if not self.gem_enable:
                if self.pacmod_enable:
                    # Configure vehicle for autonomous mode
                    self.gear_cmd.ui16_cmd = 3  # FORWARD gear
                    
                    # Enable brake control with zero brake pressure
                    self.brake_cmd.enable = True
                    self.brake_cmd.clear = False
                    self.brake_cmd.ignore = False
                    self.brake_cmd.f64_cmd = 0.0
                    
                    # Enable acceleration control with initial acceleration
                    self.accel_cmd.enable = True
                    self.accel_cmd.clear = False
                    self.accel_cmd.ignore = False
                    self.accel_cmd.f64_cmd = 1.5
                    
                    # Send initialization commands
                    self.gear_pub.publish(self.gear_cmd)
                    self.turn_pub.publish(self.turn_cmd)
                    self.brake_pub.publish(self.brake_cmd)
                    self.accel_pub.publish(self.accel_cmd)
                    
                    # Set flag indicating vehicle is now enabled
                    self.gem_enable = True
                    rospy.loginfo("GEM Enabled with Forward Gear!")

            ###############################################################################
            # Lane Following Control
            ###############################################################################
            
            # Only control if we have valid lane detection data
            if self.endgoal_x is not None:
                # Calculate lateral error (pixels from center)
                lateral_error_pixels = self.endgoal_x - self.image_center_x
                
                # Convert pixel error to steering angle (approximate mapping)
                scaling_factor = 5.0
                desired_front_angle = -lateral_error_pixels * scaling_factor
                
                ###############################################################################
                # Steering Control
                ###############################################################################
                
                # Get current time for PID controller
                current_time = rospy.get_time()
                
                # Calculate steering output using PID controller
                steering_output = self.pid_steer.get_control(current_time, lateral_error_pixels, fwd=0.0)
                
                # Convert steering output to front wheel angle
                front_angle = -steering_output * 4.0
                
                # Convert front wheel angle to steering wheel angle
                steering_angle = self.front2steer(front_angle)
                
                ###############################################################################
                # Speed Control
                ###############################################################################
                
                # Get current time for speed PID controller
                speed_time = rospy.get_time()
                
                # Calculate speed error
                speed_error = self.desired_speed - self.speed
                
                # Only adjust acceleration if speed error is significant
                if abs(speed_error) > 0.1:
                    # Calculate acceleration using PID controller
                    speed_output_accel = self.pid_speed.get_control(speed_time, speed_error)
                    
                    # Apply limits to acceleration command
                    if speed_output_accel > self.max_accel:
                        speed_output_accel = self.max_accel  # Cap maximum acceleration
                    if speed_output_accel < 0.2:
                        speed_output_accel = 0.2  # Minimum acceleration to prevent stalling
                else:
                    # Maintain current speed
                    speed_output_accel = 0.0
                
                # Update acceleration command
                self.accel_cmd.f64_cmd = speed_output_accel
                
                ###############################################################################
                # Turn Signal Control
                ###############################################################################
                
                # Set turn signals based on steering angle
                if front_angle <= 30 and front_angle >= -30:
                    self.turn_cmd.ui16_cmd = 1  # No turn signal for small steering angles
                elif front_angle > 30:
                    self.turn_cmd.ui16_cmd = 2  # Right turn signal
                else:
                    self.turn_cmd.ui16_cmd = 0  # Left turn signal
                
                # Update steering command with calculated angle (convert to radians)
                self.steer_cmd.angular_position = math.radians(steering_angle)
                
                # Log status information
                if self.gem_enable:
                    rospy.loginfo(f"Lateral error: {lateral_error_pixels} px, Steering angle: {steering_angle} deg, Speed: {self.speed} m/s")
                
                ###############################################################################
                # Publish Control Commands
                ###############################################################################
                
                # Send updated commands to vehicle
                self.accel_pub.publish(self.accel_cmd)
                self.steer_pub.publish(self.steer_cmd)
                self.turn_pub.publish(self.turn_cmd)
            
            # Wait for next control cycle
            self.rate.sleep()

###############################################################################
# Main Entry Point
###############################################################################

if __name__ == '__main__':
    # Create controller instance
    controller = LaneFollowController()
    try:
        # Start control loop
        controller.start_control()
    except rospy.ROSInterruptException:
        # Handle ROS shutdown gracefully
        pass