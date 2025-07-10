import numpy as np

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector using a quaternion rotation.
    
    Args:
        q (np.ndarray): Quaternion in [w,x,y,z] format
        v (np.ndarray): 3D vector to rotate
        
    Returns:
        np.ndarray: Rotated 3D vector
    """
    # Extract quaternion components
    w = q[0]
    x = q[1] 
    y = q[2]
    z = q[3]
    
    # Compute rotation using quaternion formula
    # v' = q * v * q_conjugate
    
    # First compute q * v (treating v as quaternion with w=0)
    t_x = w*v[0] + y*v[2] - z*v[1]
    t_y = w*v[1] + z*v[0] - x*v[2] 
    t_z = w*v[2] + x*v[1] - y*v[0]
    t_w = -x*v[0] - y*v[1] - z*v[2]
    
    # Then multiply by quaternion conjugate
    x_rot = t_w*(-x) + t_x*w + t_y*(-z) - t_z*(-y)
    y_rot = t_w*(-y) + t_y*w + t_z*(-x) - t_x*(-z)  
    z_rot = t_w*(-z) + t_z*w + t_x*(-y) - t_y*(-x)
    
    return np.array([x_rot, y_rot, z_rot])

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector using the inverse quaternion rotation.
    
    Args:
        q (np.ndarray): Quaternion in [w,x,y,z] format 
        v (np.ndarray): 3D vector to rotate
        
    Returns:
        np.ndarray: Inverse rotated 3D vector
    """
    # For unit quaternions, inverse rotation is same as rotating by conjugate
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_rotate(q_conj, v)

def euler_xyz_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (XYZ rotation sequence).
    
    Args:
        q (np.ndarray): Quaternion in [w,x,y,z] format
        
    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw] in radians
    """
    # Extract quaternion components
    w = q[0]
    x = q[1]
    y = q[2] 
    z = q[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # Use 90 degrees if out of range
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def quat_from_euler_xyz(roll, pitch, yaw) -> np.ndarray:
    """Convert Euler angles (XYZ rotation sequence) to quaternion.
    
    Args:
        euler (np.ndarray): Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        np.ndarray: Quaternion in [w,x,y,z] format
    """
    
    # Compute rotation quaternion using half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5) 
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # Quaternion multiplication for XYZ rotation sequence
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

