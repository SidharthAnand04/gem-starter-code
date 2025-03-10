import rospy
import math
import numpy as np
from std_msgs.msg import Bool
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, VehicleSpeedRpt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import time

class PID:
    """
    PID Controller implementation with anti-windup
    
    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain 
        kd (float): Derivative gain
        wg (float): Windup guard limit for integral term
        iterm (float): Integral term accumulator
        last_t (float): Time of last update
        last_e (float): Previous error
        derror (float): Current derivative error
    """
    def __init__(self, kp, ki, kd, wg=None):
        # Controller gains
        self.kp = kp
        self.ki = ki 
        self.kd = kd
        
        # Anti-windup guard
        self.wg = wg
        
        # State variables
        self.iterm = 0
        self.last_t = None
        self.last_e = 0
        self.derror = 0

    def reset(self):
        """Reset controller state variables"""
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):
        """
        Calculate control output
        
        Args:
            t (float): Current time
            e (float): Current error
            fwd (float): Feed-forward term
            
        Returns:
            float: Control output
        """
        # Calculate time delta and derivative
        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        # Reset derivative on large error changes
        if abs(e - self.last_e) > 0.5:
            de = 0

        # Update integral term
        self.iterm += e * (t - self.last_t)

        # Apply anti-windup guard to integral term
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        # Store state for next iteration
        self.last_e = e
        self.last_t = t
        self.derror = de

        # Calculate and return control output
        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de

class LaneFollowController:
    """
    Controller for autonomous lane following vehicle control.
    
    Handles vehicle control including steering, acceleration, braking and safety features.
    Uses PID controllers for speed and steering control based on lane detection input.
    """
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('lane_follow_controller', anonymous=True)
        self.rate = rospy.Rate(10)  # 10Hz control loop

        # Control parameters
        self.desired_speed = 1.5  # Target speed in m/s
        self.max_accel = 2.5      # Maximum acceleration in m/s^2
        self.image_width = 1280   # Camera image width in pixels
        self.image_center_x = self.image_width / 2.0

        # Initialize PID controllers
        self.pid_speed = PID(kp=0.5, ki=0.0, kd=0.1, wg=20)        # Speed control PID
        self.pid_steer = PID(kp=0.01, ki=0.0, kd=0.005)           # Steering control PID

        # Vehicle state variables
        self.speed = 0.0          # Current vehicle speed
        self.endgoal_x = None     # Target x position from lane detection
        self.endgoal_y = None     # Target y position from lane detection
        self.gem_enable = False   # Vehicle enable status
        self.pacmod_enable = False
        self.stop_signal = False  # Stop signal from detection

        # Set up ROS subscribers
        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)
        self.speed_sub = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.endgoal_sub = rospy.Subscriber("/lane_detection/endgoal", PoseStamped, self.endgoal_callback)
        self.stop_signal_sub = rospy.Subscriber("stop_signal/signal", Bool, self.stop_signal_callback)

        # Set up ROS publishers
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        # Initialize control commands
        self._init_control_commands()

    def _init_control_commands(self):
        """Initialize all vehicle control command messages"""
        # Enable command
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # Gear command - Forward gear
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 3

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

        # Turn signal command
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1

        # Steering command
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0
        self.steer_cmd.angular_velocity_limit = 3.5

    # Callback functions
    def enable_callback(self, msg):
        """Callback for vehicle enable status"""
        self.pacmod_enable = msg.data

    def speed_callback(self, msg):
        """Callback for vehicle speed updates"""
        self.speed = round(msg.vehicle_speed, 3)

    def endgoal_callback(self, msg):
        """Callback for lane detection target points"""
        self.endgoal_x = msg.pose.position.x
        self.endgoal_y = msg.pose.position.y

    def stop_signal_callback(self, msg):
        """Callback for stop signal detection"""
        self.stop_signal = msg.data

    def front2steer(self, f_angle):
        """
        Convert front wheel angle to steering angle
        
        Args:
            f_angle (float): Front wheel angle in degrees
        Returns:
            float: Steering angle in degrees
        """
        # Limit input angle
        f_angle = max(-35, min(35, f_angle))
        
        if f_angle > 0:
            steer_angle = round(-0.1084 * f_angle**2 + 21.775 * f_angle, 2)
        elif f_angle < 0:
            f_angle_p = -f_angle
            steer_angle = -round(-0.1084 * f_angle_p**2 + 21.775 * f_angle_p, 2)
        else:
            steer_angle = 0.0
        return steer_angle

    def start_control(self):
        """Main control loop for vehicle operation"""
        while not rospy.is_shutdown():
            # Initialize vehicle if not enabled
            if not self.gem_enable:
                self._initialize_vehicle()
                
            if self.stop_signal:
                # Handle stop signal
                self._handle_stop()
            else:
                # Normal driving mode
                if self.endgoal_x is not None:
                    self._execute_control()

            self.rate.sleep()

    def _initialize_vehicle(self):
        """Initialize vehicle when enabled"""
        if self.pacmod_enable:
            # Set initial control commands
            self.gear_cmd.ui16_cmd = 3
            
            # Enable brake control
            self.brake_cmd.enable = True
            self.brake_cmd.clear = False
            self.brake_cmd.ignore = False
            self.brake_cmd.f64_cmd = 0.0

            # Enable acceleration control
            self.accel_cmd.enable = True
            self.accel_cmd.clear = False
            self.accel_cmd.ignore = False
            self.accel_cmd.f64_cmd = 1.5

            # Publish initial commands
            self.gear_pub.publish(self.gear_cmd)
            self.turn_pub.publish(self.turn_cmd)
            self.brake_pub.publish(self.brake_cmd)
            self.accel_pub.publish(self.accel_cmd)
            
            self.gem_enable = True
            rospy.loginfo("GEM Enabled with Forward Gear!")

    def _handle_stop(self):
        """Handle vehicle stop command"""
        self.brake_cmd.f64_cmd = 1.0
        self.accel_cmd.f64_cmd = 0.0
        self.brake_pub.publish(self.brake_cmd)
        self.accel_pub.publish(self.accel_cmd)
        rospy.loginfo("Stop signal received. Stopping the vehicle.")

    def _execute_control(self):
        """Execute main control loop for steering and speed"""
        # Calculate steering control
        lateral_error_pixels = self.endgoal_x - self.image_center_x
        current_time = rospy.get_time()
        steering_output = self.pid_steer.get_control(current_time, lateral_error_pixels, fwd=0.0)
        front_angle = -steering_output * 4.0
        steering_angle = self.front2steer(front_angle)

        # Calculate speed control
        speed_error = self.desired_speed - self.speed
        speed_output_accel = self._calculate_speed_output(speed_error)

        # Update control commands
        self._update_control_commands(front_angle, steering_angle, speed_output_accel)

        # Publish control commands
        if self.gem_enable:
            rospy.loginfo(f"Lateral error: {lateral_error_pixels} px, Steering angle: {steering_angle} deg, Speed: {self.speed} m/s")
            self._publish_control_commands()

    def _calculate_speed_output(self, speed_error):
        """Calculate speed control output"""
        if abs(speed_error) > 0.1:
            speed_output = self.pid_speed.get_control(rospy.get_time(), speed_error)
            return max(0.2, min(self.max_accel, speed_output))
        return 0.0

    def _update_control_commands(self, front_angle, steering_angle, speed_output):
        """Update all control command values"""
        self.accel_cmd.f64_cmd = speed_output
        self.steer_cmd.angular_position = math.radians(steering_angle)
        
        # Update turn signal based on steering angle
        if -30 <= front_angle <= 30:
            self.turn_cmd.ui16_cmd = 1
        elif front_angle > 30:
            self.turn_cmd.ui16_cmd = 2
        else:
            self.turn_cmd.ui16_cmd = 0

    def _publish_control_commands(self):
        """Publish all control commands"""
        self.accel_pub.publish(self.accel_cmd)
        self.steer_pub.publish(self.steer_cmd)
        self.turn_pub.publish(self.turn_cmd)

if __name__ == '__main__':
    controller = LaneFollowController()
    try:
        controller.start_control()
    except rospy.ROSInterruptException:
        pass


