"""closest_point controller."""

from controller import Robot
from math import pi, cos, sin, sqrt, atan2
import numpy as np

MAX_SPEED = 12.3
INTERAXIS = 0.325
RADIUS = 0.195 / 2
Q_START = np.array([-1.0, 2.0, pi])
Q_SENSOR = np.array([0.08, 0, 0])
LIDAR_RANGE = pi
DIST_FROM_WALL = 0.5 - Q_SENSOR[0]
Q_INTERM = np.array([-1.0, 2.0, 0])
K = np.array([0.25, 0.25, 0.5])
TOL = 0.01

class Controller:
    def __init__(self, robot, timestep, interaxis, radius, max_speed, q_start, q_sensor, range, k, min_dist, tol):
        self.robot = robot
        self.timestep = timestep
        self.interaxis = interaxis
        self.radius = radius
        self.max_speed = max_speed
        self.x_robot = q_start[0]
        self.y_robot = q_start[1]
        self.theta_robot = q_start[2]
        self.q_robot = np.array([self.x_robot, self.y_robot, self.theta_robot])
        self.q_sensor = q_sensor
        self.range = range
        self.k = k
        self.min_dist = min_dist
        self.tol = tol
        self.lidar = robot.getDevice('Sick LMS 291')
        self.lidar.enable(60)
        self.lidar.enablePointCloud()
        self.x_goal = 0
        self.y_goal = 0
        self.theta_goal = 0
        self.q_goal = np.array([self.x_goal, self.y_goal, self.theta_goal])
        self.x_robot_goal = 0
        self.y_robot_goal = 0
        self.theta_robot_goal = 0
        self.q_robot_goal = np.array([self.x_robot_goal, self.y_robot_goal, self.theta_robot_goal])
        self.current_time = 0
        self.start_time = 0
        self.linear_velocity = 0
        self.angular_velocity = 0
        self.left_wheel_velocity = 0
        self.right_wheel_velocity = 0
        self.curvature_radius = 0
        self.x_velocity_centre = 0
        self.y_velocity_centre = 0
        self.near_obstacle = (0, 0)
        self.transform_velocities = np.array([[1/self.radius, -self.interaxis/(2*self.radius)],
                                              [1/self.radius, self.interaxis/(2*self.radius)]])
        self.inverse_transform_velocities = np.linalg.inv(self.transform_velocities)
        self.leftMotor = self.robot.getDevice('left wheel')
        self.rightMotor = self.robot.getDevice('right wheel')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

    def transform_pos_direct(self, q_ref, q_in_matrix):
        q_in_matrix = q_in_matrix.copy()
        if len(q_in_matrix.shape) == 1:
            q_in_matrix = q_in_matrix.reshape(3, 1)
        transl = np.array([q_ref[0], q_ref[1], q_ref[2]])
        rot = np.array([[cos(q_ref[2]), -sin(q_ref[2]), 0],
                        [sin(q_ref[2]), cos(q_ref[2]), 0],
                        [0, 0, 1]])
        return np.apply_along_axis(lambda x: transl + rot @ x, 0, q_in_matrix).ravel()

    def transform_pos_inverse(self, q_ref, q_in_matrix):
        q_in_matrix = q_in_matrix.copy()
        if len(q_in_matrix.shape) == 1:
            q_in_matrix = q_in_matrix.reshape(3, 1)
        transl = np.array([q_ref[0], q_ref[1], q_ref[2]])
        rot = np.array([[cos(q_ref[2]), sin(q_ref[2]), 0],
                        [-sin(q_ref[2]), cos(q_ref[2]), 0],
                        [0, 0, 1]])
        return np.apply_along_axis(lambda x: rot @ (x - transl), 0, q_in_matrix).ravel()

    def calc_velocity_centre(self):
        if self.angular_velocity == 0:
            self.curvature_radius = 0
            self.x_velocity_centre = 0
            self.y_velocity_centre = 0
        else:
            self.curvature_radius = self.linear_velocity / self.angular_velocity
            self.x_velocity_centre = self.x_robot - self.curvature_radius * sin(self.theta_robot)
            self.y_velocity_centre = self.y_robot + self.curvature_radius * cos(self.theta_robot)

    def state_update(self):
        self.current_time += self.timestep
        self.calc_velocity_centre()
        if self.angular_velocity == 0:
            self.x_robot += self.linear_velocity * cos(self.theta_robot) * self.timestep
            self.y_robot += self.linear_velocity * sin(self.theta_robot) * self.timestep
        else:
            xy_robot = self.q_robot[:2]
            velocity_centre = np.array([self.x_velocity_centre, self.y_velocity_centre])
            rot_matrix = np.array([[cos(self.angular_velocity * self.timestep), -sin(self.angular_velocity * self.timestep)],
                                   [sin(self.angular_velocity * self.timestep), cos(self.angular_velocity * self.timestep)]])
            xy_robot_new = velocity_centre + rot_matrix @ (xy_robot - velocity_centre)
            self.x_robot = xy_robot_new[0]
            self.y_robot = xy_robot_new[1]
            self.theta_robot += self.angular_velocity * self.timestep
        self.q_robot = np.array([self.x_robot, self.y_robot, self.theta_robot])
        self.q_robot_goal = self.transform_pos_inverse(self.q_goal, self.q_robot)
        self.x_robot_goal = self.q_robot_goal[0]
        self.y_robot_goal = self.q_robot_goal[1]
        self.theta_robot_goal = self.q_robot_goal[2]
        # print("Current time: %g s" % (self.current_time))
        # print("Current state robot: x = %g m; y = %g m; theta = %g rad" % (self.x_robot, self.y_robot, self.theta_robot))
        # print("Current state robot_goal: x = %g m; y = %g m; theta = %g rad" % (self.x_robot_goal, self.y_robot_goal, self.theta_robot_goal))
        self.near_obstacle = self.min_distance_from_obstacle()

    def move_robot(self, linear_velocity, angular_velocity):
        robot_velocities = np.array([linear_velocity, angular_velocity])
        wheel_velocities = self.transform_velocities @ robot_velocities
        max_speed = max(abs(wheel_velocities[0]), abs(wheel_velocities[1]))
        if max_speed > self.max_speed:
            scale_factor = self.max_speed / max_speed
            robot_velocities *= scale_factor
            wheel_velocities *= scale_factor
        self.linear_velocity = robot_velocities[0]
        self.angular_velocity = robot_velocities[1]
        self.left_wheel_velocity = wheel_velocities[0]
        self.right_wheel_velocity = wheel_velocities[1]
        self.leftMotor.setVelocity(self.left_wheel_velocity)
        self.rightMotor.setVelocity(self.right_wheel_velocity)
        # print("Set robot velocities to: %g m/s linear, %g rad/s angular" %
        #       (self.linear_velocity, self.angular_velocity))

    def set_goal_q(self, q_goal):
        self.q_goal = q_goal
        self.x_goal = q_goal[0]
        self.y_goal = q_goal[1]
        self.theta_goal = q_goal[2]
    
    def set_goal_track(self, y_goal, theta_goal):
        self.x_goal = 0
        self.y_goal = y_goal
        self.theta_goal = theta_goal
        self.q_goal = np.array([self.x_goal, self.y_goal, self.theta_goal])

    def control_to_goal_q(self):
        linear_velocity = - self.k[0] * self.x_robot_goal
        angular_velocity = - self.k[1] * self.y_robot_goal - self.k[2] * self.theta_robot_goal
        self.move_robot(linear_velocity, angular_velocity)

    def control_to_goal_track(self, linear_velocity):
        angular_velocity = - self.k[1] * self.y_robot_goal - self.k[2] * self.theta_robot_goal
        self.move_robot(linear_velocity, angular_velocity)

    def reached_goal_q(self):
        error = np.linalg.norm(self.q_robot_goal[:2]) + abs(self.theta_robot_goal)
        if error < self.tol:
            return True
        else:
            return False

    def reached_goal_track(self):
        error = abs(self.y_robot_goal) + abs(self.theta_robot_goal)
        if error < self.tol:
            return True
        else:
            return False

    def min_distance_from_obstacle(self):
        scan = np.array(self.lidar.getRangeImage())
        angles = np.linspace(self.range / 2, - self.range / 2, scan.shape[0])
        index = np.argmin(scan)
        min_angle = angles[index]
        min_distance = scan[index]
        return (min_angle, min_distance)

    def plan_to_wall(self):
        distance = self.near_obstacle[1] - self.min_dist
        angle = self.near_obstacle[0]
        q_wall_sensor = np.array([distance * cos(angle), distance * sin(angle), angle])
        q_wall_robot = self.transform_pos_direct(self.q_sensor, q_wall_sensor)
        return self.transform_pos_direct(self.q_robot, q_wall_robot)

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    ctrl = Controller(robot, timestep / 1000, INTERAXIS, RADIUS, MAX_SPEED, Q_START, Q_SENSOR, LIDAR_RANGE, K, DIST_FROM_WALL, TOL)
    print("Current time: %g s" % (ctrl.current_time))
    print("Current state: x = %g m; y = %g m; theta = %g rad" %
        (ctrl.x_robot, ctrl.y_robot, ctrl.theta_robot))

    ctrl.set_goal_q(Q_INTERM)
    while robot.step(timestep) != -1:
        ctrl.state_update()
        ctrl.control_to_goal_q()
        if ctrl.reached_goal_q():
            break
    print("Current time: %g s" % (ctrl.current_time))
    print("Current state: x = %g m; y = %g m; theta = %g rad" %
        (ctrl.x_robot, ctrl.y_robot, ctrl.theta_robot))

    print("Min distance from obstacle: %g m at angle %g rad" % (ctrl.near_obstacle[1], ctrl.near_obstacle[0]))

    ctrl.set_goal_q(ctrl.plan_to_wall())
    while robot.step(timestep) != -1:
        ctrl.state_update()
        ctrl.control_to_goal_q()
        if ctrl.reached_goal_q():
            break
    print("Current time: %g s" % (ctrl.current_time))
    print("Current state: x = %g m; y = %g m; theta = %g rad" %
        (ctrl.x_robot, ctrl.y_robot, ctrl.theta_robot))
    ctrl.move_robot(0.0, 0.0)

if __name__ == "__main__":
    main()
