import position_control as pc

import numpy as np

if __name__ == "__main__":
    # Setup the position controller
    ctrl = pc.PositionControl()
    ctrl.set_position_gains(np.array([0.95, 0.95, 1.0]))
    ctrl.set_velocity_gains(
        p_gains=np.array([1.8, 1.8, 4]),
        d_gains=np.array([0.2, 0.2, 0.0]),
        i_gains=np.array([0.4, 0.4, 2.0]))
    ctrl.set_velocity_limits(
        vel_horizontal=np.array([10.0, 10.0]),
        vel_up=5.0,
        vel_down=-3.0)
    ctrl.set_thrust_limits(
        min_thrust=0.1,
        max_thrust=1.0)
    ctrl.set_horizontal_thrust_margin(0.3)
    ctrl.set_tilt_limit(np.pi/4.0)
    ctrl.set_hover_thrust(0.7)

    # Set commands
    pc_states = pc.PositionControlStates(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        acceleration=np.array([0.0, 0.0, 0.0]),
        yaw=0.0,
    )
    ctrl.set_state(pc_states)

    pc_setpoint = pc.TrajectorySetpoint(
        timestamp=0.0,
        position=np.array([np.nan, np.nan, np.nan]),
        velocity=np.array([0.0, 1.0, 0.0]),
        acceleration=np.array([0.0, 0.0, 0.0]),
        jerk=np.array([np.nan, np.nan, np.nan]),
        yaw=0.0,
        yawspeed=0.0,
    )
    ctrl.set_input_setpoint(pc_setpoint)
    
    # Run
    dt = 0.01
    valid = ctrl.update(dt)
    print(f"Valid: {valid}")
    local_position_setpoint = ctrl.get_local_position_setpoint()
    print(f"Position setpoint: {local_position_setpoint[0]}")
    print(f"Velocity setpoint: {local_position_setpoint[1]}")
    print(f"Acceleration setpoint: {local_position_setpoint[2]}")
    print(f"Yaw setpoint: {local_position_setpoint[3]}")
    print(f"Yaw speed setpoint: {local_position_setpoint[4]}")
    att_sp: pc.VehicleAttitudeSetpoint = ctrl.get_attitude_setpoint()
    print(f"Roll setpoint: {att_sp.roll_body}")
    print(f"Pitch setpoint: {att_sp.pitch_body}")
    print(f"Thrust setpoint: {att_sp.thrust_body}")
