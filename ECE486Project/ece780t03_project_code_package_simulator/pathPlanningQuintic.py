import numpy as np
import matplotlib.pyplot as plt

def quintic_spline(t, start, end):
    """
    Generate quintic spline coefficients.
    """
    t = t - t[0]  # Normalize time to start at 0
    T = t[-1]     # Total duration
    
    # Quintic polynomial coefficients
    A = np.array([
        [0, 0, 0, 0, 0, 1],
        [T**5, T**4, T**3, T**2, T, 1],
        [0, 0, 0, 0, 1, 0],
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]
    ])
    
    b = np.array([
        start[0], end[0],  # Position
        start[1], end[1],  # Velocity
        start[2], end[2]   # Acceleration
    ])
    
    coeffs = np.linalg.solve(A, b)
    return coeffs

def evaluate_quintic(t, coeffs):
    """
    Evaluate quintic spline and its derivatives at given time points.
    """
    t = t[:, np.newaxis]
    power_matrix = t**np.arange(5, -1, -1)
    
    # Position
    pos = np.sum(coeffs * power_matrix, axis=1)
    
    # Velocity
    vel_coeffs = coeffs[:-1] * np.arange(5, 0, -1)
    vel = np.sum(vel_coeffs * power_matrix[:, 1:], axis=1)
    
    # Acceleration
    acc_coeffs = vel_coeffs[:-1] * np.arange(4, 0, -1)
    acc = np.sum(acc_coeffs * power_matrix[:, 2:], axis=1)
    
    return pos, vel, acc

def quintic_spline_path_planning(start_pose, end_pose, duration=10, num_points=1000):
    x_start, y_start, theta_start = start_pose
    x_end, y_end, theta_end = end_pose

    # Create time array
    t = np.linspace(0, duration, num_points)

    # Generate quintic splines for x and y
    start_x = [x_start, np.cos(theta_start), 0]  # position, velocity, acceleration
    end_x = [x_end, np.cos(theta_end)*0.6, 0]
    coeffs_x = quintic_spline(t, start_x, end_x)

    start_y = [y_start, np.sin(theta_start), 0]
    end_y = [y_end, np.sin(theta_end)*0.6, 0]
    coeffs_y = quintic_spline(t, start_y, end_y)

    # Evaluate splines
    x, x_dot, x_ddot = evaluate_quintic(t, coeffs_x)
    y, y_dot, y_ddot = evaluate_quintic(t, coeffs_y)

    # Calculate linear velocity v(t)
    v = np.sqrt(x_dot**2 + y_dot**2)

    # Calculate angular velocity ω(t)
    omega = (y_ddot * x_dot - x_ddot * y_dot) / (x_dot**2 + y_dot**2)

    # Calculate heading angle theta
    theta = np.arctan2(y_dot, x_dot)

    return t, x, y, theta, v, omega

def plot_results(t, x, y, theta, v, omega, start_pose, end_pose):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot path
    axs[0, 0].plot(x, y, 'b-', label='Path')
    axs[0, 0].plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    axs[0, 0].plot(end_pose[0], end_pose[1], 'ro', markersize=10, label='End')
    axs[0, 0].arrow(start_pose[0], start_pose[1], np.cos(start_pose[2]), np.sin(start_pose[2]), color='g', width=0.1)
    axs[0, 0].arrow(end_pose[0], end_pose[1], np.cos(end_pose[2]), np.sin(end_pose[2]), color='r', width=0.1)
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].set_title('Unicycle Path')
    axs[0, 0].legend()
    axs[0, 0].axis('equal')
    axs[0, 0].grid(True)

    # Plot x(t) and y(t)
    axs[0, 1].plot(t, x, 'r-', label='x(t)')
    axs[0, 1].plot(t, y, 'b-', label='y(t)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Position')
    axs[0, 1].set_title('Position vs Time')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot theta(t)
    axs[1, 0].plot(t, theta, 'g-', label='θ(t)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Heading Angle (rad)')
    axs[1, 0].set_title('Heading Angle vs Time')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    # Plot v(t)
    axs[1, 1].plot(t, v, 'm-', label='v(t)')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Linear Velocity')
    axs[1, 1].set_title('Linear Velocity vs Time')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Plot ω(t)
    axs[2, 0].plot(t, omega, 'c-', label='ω(t)')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Angular Velocity (rad/s)')
    axs[2, 0].set_title('Angular Velocity vs Time')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    #plt.tight_layout()
    plt.show()

# Example usage


start_pose = (-2, -2, 0)  # (x, y, theta)
end_pose = (1.5, 2.5, -np.pi/6)  # (x, y, theta)

t, x, y, theta, v, omega = quintic_spline_path_planning(start_pose, end_pose, duration=10)
plot_results(t, x, y, theta, v, omega, start_pose, end_pose)