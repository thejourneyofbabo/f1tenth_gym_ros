#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
import numpy as np
import math

class SensorNoiseNode(Node):
    def __init__(self):
        super().__init__('sensor_noise_node')
        
        # Declare parameters
        self.declare_parameter('input_scan_topic', '/scan')
        self.declare_parameter('input_odom_topic', '/ego_racecar/odom')
        self.declare_parameter('output_scan_topic', '/scan_noisy')
        self.declare_parameter('output_odom_topic', '/ego_racecar/odom_noisy')
        self.declare_parameter('lidar_noise_scale', 1.0)
        self.declare_parameter('odom_noise_scale', 1.0)
        
        # Get parameters
        self.input_scan_topic = self.get_parameter('input_scan_topic').value
        self.input_odom_topic = self.get_parameter('input_odom_topic').value
        self.output_scan_topic = self.get_parameter('output_scan_topic').value
        self.output_odom_topic = self.get_parameter('output_odom_topic').value
        self.lidar_noise_scale = self.get_parameter('lidar_noise_scale').value
        self.odom_noise_scale = self.get_parameter('odom_noise_scale').value
        
        # Create publishers and subscribers
        self.scan_pub = self.create_publisher(LaserScan, self.output_scan_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.output_odom_topic, 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan, self.input_scan_topic, self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.input_odom_topic, self.odom_callback, 10)
        
        self.get_logger().info(f'Sensor noise node initialized')
        self.get_logger().info(f'Input scan: {self.input_scan_topic} -> Output scan: {self.output_scan_topic}')
        self.get_logger().info(f'Input odom: {self.input_odom_topic} -> Output odom: {self.output_odom_topic}')
        self.get_logger().info(f'Lidar noise scale: {self.lidar_noise_scale}')
        self.get_logger().info(f'Odom noise scale: {self.odom_noise_scale}')
    
    def scan_callback(self, msg):
        """Add realistic noise to laser scan data"""
        noisy_scan = LaserScan()
        noisy_scan.header = msg.header
        noisy_scan.angle_min = msg.angle_min
        noisy_scan.angle_max = msg.angle_max
        noisy_scan.angle_increment = msg.angle_increment
        noisy_scan.time_increment = msg.time_increment
        noisy_scan.scan_time = msg.scan_time
        noisy_scan.range_min = msg.range_min
        noisy_scan.range_max = msg.range_max
        
        # Add realistic noise to ranges
        noisy_ranges = []
        for range_val in msg.ranges:
            if math.isinf(range_val) or math.isnan(range_val):
                noisy_ranges.append(range_val)
            else:
                # Add Gaussian noise with standard deviation proportional to range
                # This models the fact that laser range finders have higher uncertainty at longer ranges
                noise_std = 0.02 + 0.01 * range_val  # 2cm + 1% of range
                noise = np.random.normal(0, noise_std * self.lidar_noise_scale)
                noisy_range = range_val + noise
                
                # Ensure the noisy range is within valid bounds
                noisy_range = max(msg.range_min, min(msg.range_max, noisy_range))
                noisy_ranges.append(noisy_range)
        
        noisy_scan.ranges = noisy_ranges
        noisy_scan.intensities = msg.intensities
        
        self.scan_pub.publish(noisy_scan)
    
    def odom_callback(self, msg):
        """Add realistic noise to odometry data"""
        noisy_odom = Odometry()
        noisy_odom.header = msg.header
        noisy_odom.child_frame_id = msg.child_frame_id
        
        # Add noise to pose
        pose = msg.pose.pose
        noisy_pose = PoseWithCovariance()
        noisy_pose.pose = pose
        
        # Add noise to position (x, y)
        pos_noise_std = 0.01 * self.odom_noise_scale  # 1cm standard deviation
        noisy_pose.pose.position.x = pose.position.x + np.random.normal(0, pos_noise_std)
        noisy_pose.pose.position.y = pose.position.y + np.random.normal(0, pos_noise_std)
        noisy_pose.pose.position.z = pose.position.z
        
        # Add noise to orientation (yaw)
        # Convert quaternion to euler angles, add noise to yaw, convert back
        yaw = self.quaternion_to_yaw(pose.orientation)
        yaw_noise_std = 0.01 * self.odom_noise_scale  # ~0.6 degrees standard deviation
        noisy_yaw = yaw + np.random.normal(0, yaw_noise_std)
        noisy_pose.pose.orientation = self.yaw_to_quaternion(noisy_yaw)
        
        # Set reasonable covariance values
        noisy_pose.covariance = [
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0,      # x variance
            0.0, 0.01, 0.0, 0.0, 0.0, 0.0,      # y variance
            0.0, 0.0, 0.01, 0.0, 0.0, 0.0,      # z variance
            0.0, 0.0, 0.0, 0.01, 0.0, 0.0,      # roll variance
            0.0, 0.0, 0.0, 0.0, 0.01, 0.0,      # pitch variance
            0.0, 0.0, 0.0, 0.0, 0.0, 0.01       # yaw variance
        ]
        
        noisy_odom.pose = noisy_pose
        
        # Add noise to twist (velocity)
        twist = msg.twist.twist
        noisy_twist = TwistWithCovariance()
        noisy_twist.twist = twist
        
        # Add noise to linear velocity
        vel_noise_std = 0.05 * self.odom_noise_scale  # 5cm/s standard deviation
        noisy_twist.twist.linear.x = twist.linear.x + np.random.normal(0, vel_noise_std)
        noisy_twist.twist.linear.y = twist.linear.y + np.random.normal(0, vel_noise_std)
        noisy_twist.twist.linear.z = twist.linear.z
        
        # Add noise to angular velocity
        ang_vel_noise_std = 0.05 * self.odom_noise_scale  # ~3 degrees/s standard deviation
        noisy_twist.twist.angular.x = twist.angular.x + np.random.normal(0, ang_vel_noise_std)
        noisy_twist.twist.angular.y = twist.angular.y + np.random.normal(0, ang_vel_noise_std)
        noisy_twist.twist.angular.z = twist.angular.z + np.random.normal(0, ang_vel_noise_std)
        
        # Set reasonable covariance values for twist
        noisy_twist.covariance = [
            0.01, 0.0, 0.0, 0.0, 0.0, 0.0,      # linear.x variance
            0.0, 0.01, 0.0, 0.0, 0.0, 0.0,      # linear.y variance
            0.0, 0.0, 0.01, 0.0, 0.0, 0.0,      # linear.z variance
            0.0, 0.0, 0.0, 0.01, 0.0, 0.0,      # angular.x variance
            0.0, 0.0, 0.0, 0.0, 0.01, 0.0,      # angular.y variance
            0.0, 0.0, 0.0, 0.0, 0.0, 0.01       # angular.z variance
        ]
        
        noisy_odom.twist = noisy_twist
        
        self.odom_pub.publish(noisy_odom)
    
    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = SensorNoiseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 