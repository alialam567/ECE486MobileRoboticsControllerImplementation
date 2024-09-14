from mobile_manipulator_unicycle_sim import MobileManipulatorUnicycleSim as Sim
from pathPlanningQuintic import quintic_spline_path_planning
import time
import math
import numpy as np
import matplotlib.pyplot as plt

start_pose = [-2, -2, 0]  # (x, y, theta)
end_pose = [1.5, 2.5, -np.pi/6]  # (x, y, theta)

robot = Sim(robot_id=1,
            robot_pose=start_pose,
            pickup_location=[-2,-2],
            dropoff_location=[1.5, 2.5],
            obstacles_location=[[3, 3], [3,3]])

start_time = time.time()
t, x, y, theta, v, omega = quintic_spline_path_planning(start_pose, end_pose, duration=10, num_points=1000)
while time.time() - start_time < 10.:
    currentTime = math.floor((time.time() - start_time)*100)
    robot.set_mobile_base_speed_and_gripper_power(v=v[currentTime], omega=omega[currentTime], gripper_power=0.0)

# Stop the drive base and the gripper.
robot.set_mobile_base_speed_and_gripper_power(0., 0., 0.)

# Get the robot's current pose.
poses = robot.get_poses()
print(f"Robot, pickup, dropoff, obstacles poses: {poses}")
