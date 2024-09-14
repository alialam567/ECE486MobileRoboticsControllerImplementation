import numpy as np
import matplotlib.pyplot as plt
import time
import math
from mobile_manipulator_unicycle_sim import MobileManipulatorUnicycleSim as Sim
from pathPlanningQuintic import quintic_spline_path_planning

def state_feedback_control(desired_state, current_state, Kp, Kd):
    error = desired_state[:2] - current_state[:2]
    error_dot = desired_state[2:] - current_state[2:]
    uw = Kp * error + Kd * error_dot
    return uw

def run_simulation_with_feedback(start_pose, end_pose, Kp, Kd, num_simulations=100):
    duration = 1000
    num_points = 1000

    # Plan the path
    t, x, y, theta, v, omega = quintic_spline_path_planning(start_pose, end_pose, duration, num_points)
    
    # Run simulations
    all_trajectories = []
    for _ in range(num_simulations):
        # Initialize the robot simulation
        initial_pose = np.array(start_pose) + np.random.normal(0, 0.1, 3)
        robot = Sim(robot_id=1, robot_pose=initial_pose.tolist(),
                    pickup_location=[-2, -2], dropoff_location=[1.5, 2.5],
                    obstacles_location=[[3, 3], [3, 3]])
        
        start_time = time.time()
        trajectory = []

        prev_v = 0
        prev_w = 0

        while time.time() - start_time < duration:
            current_time = math.floor((time.time() - start_time) * (num_points / duration))
            
            if current_time >= num_points:
                break

            desired_state = np.array([x[current_time], y[current_time], v[current_time], omega[current_time]])
            current_pose = robot.get_poses()[0]  # Get the robot's current pose
            current_state = np.array([current_pose[0], current_pose[1], prev_v, prev_w])
            
            uw = state_feedback_control(desired_state, current_state, Kp, Kd)
            
            robot.set_mobile_base_speed_and_gripper_power(v=uw[0], omega=uw[1], gripper_power=0.0)
            prev_v=uw[0]
            prev_w=uw[1]
            trajectory.append(current_pose[:2])
        
        # Stop the robot
        robot.set_mobile_base_speed_and_gripper_power(0., 0., 0.)
        
        all_trajectories.append(np.array(trajectory))
    
    return t, all_trajectories

def plot_simulation_results(all_trajectories, start_pose, end_pose):
    plt.figure(figsize=(8, 8))
    for trajectory in all_trajectories:
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.1)
    plt.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    plt.plot(end_pose[0], end_pose[1], 'ro', markersize=10, label='End')
    plt.arrow(start_pose[0], start_pose[1], np.cos(start_pose[2]), np.sin(start_pose[2]), color='g', width=0.1)
    plt.arrow(end_pose[0], end_pose[1], np.cos(end_pose[2]), np.sin(end_pose[2]), color='r', width=0.1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories with State Feedback Control')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Parameters
start_pose = [-2, -2, 0]  # (x, y, theta)
end_pose = [1.5, 2.5, -np.pi/6]  # (x, y, theta)
Kp = np.array([.10, 10.0])
Kd = np.array([1, .1])

# Run simulations
t, all_trajectories = run_simulation_with_feedback(start_pose, end_pose, Kp, Kd)

# Plot the results
plot_simulation_results(all_trajectories, start_pose, end_pose)