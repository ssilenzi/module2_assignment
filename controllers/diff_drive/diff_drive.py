"""diff_drive controller."""

from controller import Robot
from math import pi, cos, sin, sqrt, atan2
import numpy as np

MAX_SPEED = 12.3
INTERAXIS = 0.325
RADIUS = 0.195 / 2
Q_START = np.array([-1.0, 2.0, pi])
Q_GOAL = np.array([-1.0, -2.0, pi])

Q_INTERM = np.array([-1.0, -2.0, 2*pi])
TOL = 0.01


class Controller:
    def __init__(self, robot, timestep, interaxis, radius, max_speed, q_start):
        self.robot = robot
        self.timestep = timestep
        self.interaxis = interaxis
        self.radius = radius
        self.max_speed = max_speed
        self.x = q_start[0]
        self.y = q_start[1]
        self.theta = q_start[2]
        self.leftMotor = self.robot.getDevice('left wheel')
        self.rightMotor = self.robot.getDevice('right wheel')
        self.current_time = 0
        self.start_time = 0
        self.linear_velocity = 0
        self.angular_velocity = 0
        self.left_wheel_velocity = 0
        self.right_wheel_velocity = 0
        self.curvature_radius = 0
        self.x_velocity_centre = 0
        self.y_velocity_centre = 0
        self.transform_velocities = np.array([[1/self.radius, -self.interaxis/(2*self.radius)],
                                              [1/self.radius, self.interaxis/(2*self.radius)]])
        self.inverse_transform_velocities = np.linalg.inv(
            self.transform_velocities)
        # the control here is in velocity, so no sense in having a position sensor
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

    def calc_velocity_centre(self):
        if self.angular_velocity == 0:
            self.curvature_radius = 0
            self.x_velocity_centre = 0
            self.y_velocity_centre = 0
        else:
            self.curvature_radius = self.linear_velocity / self.angular_velocity
            self.x_velocity_centre = self.x - \
                self.curvature_radius * sin(self.theta)
            self.y_velocity_centre = self.y + \
                self.curvature_radius * cos(self.theta)

    def state_update(self):
        self.current_time += self.timestep
        self.calc_velocity_centre()
        xy = np.array([self.x, self.y])
        if self.angular_velocity == 0:
            self.x += self.linear_velocity * cos(self.theta) * self.timestep
            self.y += self.linear_velocity * sin(self.theta) * self.timestep
        else:
            velocity_centre = np.array(
                [self.x_velocity_centre, self.y_velocity_centre])
            rot_matrix = np.array([[cos(self.angular_velocity * self.timestep), -sin(self.angular_velocity * self.timestep)],
                                   [sin(self.angular_velocity * self.timestep), cos(self.angular_velocity * self.timestep)]])
            xy_new = velocity_centre + rot_matrix @ (xy - velocity_centre)
            self.x = xy_new[0]
            self.y = xy_new[1]
            self.theta += self.angular_velocity * self.timestep

    def move_robot(self, linear, angular):
        robot_velocities = np.array([linear, angular])
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
        print("Set robot velocities to: %g m/s linear, %g rad/s angular" %
              (self.linear_velocity, self.angular_velocity))

    def set_position(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def goal_reached(self, q_goal, tol):
        if np.linalg.norm(q_goal[0:2] - np.array([self.x, self.y])) + abs(q_goal[2] - self.theta) < tol:
            return True
        else:
            return False

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    ctrl = Controller(robot, timestep / 1000, INTERAXIS, RADIUS, MAX_SPEED, Q_START)
    print("Current time: %g s" % (ctrl.current_time))
    print("Current state: x = %g m; y = %g m; theta = %g rad" %
        (ctrl.x, ctrl.y, ctrl.theta))

    ctrl.move_robot(0.5, 0.25)
    while robot.step(timestep) != -1:
        ctrl.state_update()
        if ctrl.goal_reached(Q_INTERM, TOL):
            break
    print("Current time: %g s" % (ctrl.current_time))
    print("Current state: x = %g m; y = %g m; theta = %g rad" %
        (ctrl.x, ctrl.y, ctrl.theta))

    ctrl.move_robot(0, -0.25)
    while robot.step(timestep) != -1:
        ctrl.state_update()
        if ctrl.goal_reached(Q_GOAL, TOL):
            break
    print("Current time: %g s" % (ctrl.current_time))
    print("Current state: x = %g m; y = %g m; theta = %g rad" %
        (ctrl.x, ctrl.y, ctrl.theta))
    ctrl.move_robot(0.0, 0.0)

if __name__ == "__main__":
    main()
