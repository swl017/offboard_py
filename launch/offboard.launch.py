from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def get_config(context):
    config_file = LaunchConfiguration('config_file').perform(context)
    
    # If absolute path, use it directly
    if os.path.isabs(config_file):
        vehicles_config_path = config_file
    else:
        # If relative path, look in the package's share directory
        package_share_dir = get_package_share_directory('offboard_py')
        vehicles_config_path = os.path.join(package_share_dir, config_file)
    
    if not os.path.exists(vehicles_config_path):
        raise FileNotFoundError(f"Vehicles config file not found: {vehicles_config_path}")
        
    with open(vehicles_config_path, 'r') as f:
        vehicles_config = yaml.safe_load(f)
        
    return vehicles_config

def launch_setup(context):
    vehicles_config = get_config(context)
    nodes = []
    
    for vehicle in vehicles_config['vehicles']:
        # vehicle_name = vehicle['name']
        namespace = vehicle['namespace']
        
        node = Node(
            package='offboard_py',
            executable='offboard_control',
            name='offboard_control',
            # name=f'offboard_control_{vehicle_name}',
            namespace=namespace,
            output='screen',
            emulate_tty=True,
            parameters=[{
                'vehicle_name': namespace,
                'update_rate': vehicle.get('update_rate', 100.0),
                'target_system': vehicle.get('target_system', 1)
            }],
            remappings=[
                # (f'fmu/in/offboard_control_mode', f'{namespace}/fmu/in/offboard_control_mode'),
                # (f'fmu/in/trajectory_setpoint', f'{namespace}/fmu/in/trajectory_setpoint'),
                # (f'fmu/in/vehicle_command', f'{namespace}/fmu/in/vehicle_command'),
                # (f'fmu/out/vehicle_local_position', f'{namespace}/fmu/out/vehicle_local_position'),
                # (f'fmu/out/vehicle_status', f'{namespace}/fmu/out/vehicle_status')
            ]
        )
        nodes.append(node)
    
    return nodes

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value='config/vehicles.yaml',
            description='Path to the vehicle configuration file'
        ),
        OpaqueFunction(function=launch_setup)
    ])