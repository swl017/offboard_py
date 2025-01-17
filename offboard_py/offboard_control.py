#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from geometry_msgs.msg import TwistStamped

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control')

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('vehicle_name', ''),
                ('update_rate', 100.0),
                ('target_system', 1)
            ]
        )
        
        # Get parameters
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.update_rate = self.get_parameter('update_rate').value
        self.target_system = self.get_parameter('target_system').value
        
        # Log parameter values and indicate if using defaults
        self.get_logger().info(f'Initialized offboard control with parameters:')
        self.get_logger().info(f'  - Vehicle name: {self.vehicle_name}' + 
                             (' (default)' if self.vehicle_name == '' else ''))
        self.get_logger().info(f'  - Update rate: {self.update_rate} Hz' +
                             (' (default)' if self.update_rate == 100.0 else ''))
        self.get_logger().info(f'  - Target system: {self.target_system}' +
                             (' (default)' if self.target_system == 1 else ''))

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, 'fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, 'fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, 'fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_cmd_vel_subscriber = self.create_subscription(
            TwistStamped, 'cmd_vel', self.vehicle_cmd_vel_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -1.0
        self.vehicle_cmd_vel = TwistStamped()
        self.pass_vehicle_cmd_vel = False

        # Create a timer to publish control commands
        timer_period = 1.0 / self.update_rate  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def vehicle_cmd_vel_callback(self, vehcile_cmd_vel):
        self.vehcile_cmd_vel = vehcile_cmd_vel
        if self.pass_vehicle_cmd_vel:
            self.publish_velocity_setpoint(self.vehcile_cmd_vel.twist.linear.x,
                                           self.vehcile_cmd_vel.twist.linear.y,
                                           self.vehcile_cmd_vel.twist.linear.z,
                                           self.vehcile_cmd_vel.twist.angular.z)

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f'{self.vehicle_name}: Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info(f'{self.vehicle_name}: Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(f'{self.vehicle_name}: Switching to offboard mode')

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info(f'{self.vehicle_name}: Switching to land mode')

    def publish_offboard_control_heartbeat_signal(self, mode: str) -> None:
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 1.57079  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_velocity_setpoint(self, vx_enu: float, vy_enu: float, vz_enu: float, vyaw: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.velocity = [vy_enu, vx_enu, -vz_enu]
        msg.yawspeed = -vyaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = self.target_system
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal('velocity')

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height \
            and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD \
            and not self.pass_vehicle_cmd_vel:
            # self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)
            """Publish the trajectory setpoint."""
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), float('nan')]
            msg.velocity = [0.0, 0.0, -1.0]
            msg.yawspeed = 0.0
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)
            if self.vehicle_local_position.z <= self.takeoff_height * 0.7:
                self.pass_vehicle_cmd_vel = True

        elif self.pass_vehicle_cmd_vel:
            # self.publish_offboard_control_heartbeat_signal('velocity')
            """Publish the trajectory setpoint."""
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), -2.0]
            msg.velocity = [0.0, -1.0, 0.0]
            msg.yawspeed = 0.0
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_publisher.publish(msg)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)