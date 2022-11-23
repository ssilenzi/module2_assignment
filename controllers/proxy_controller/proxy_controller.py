"""proxy_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

MAX_SPEED = 12.3

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# get wheel motor controllers
leftMotor = robot.getDevice('left wheel')
rightMotor = robot.getDevice('right wheel')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# set wheel velocities (in rad)
# a velocity of 2*pi means that the wheel will make one turn per second
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# get and enable lidar
lidar = robot.getDevice('Sick LMS 291')
lidar.enable(60)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # get current lidar measurements
    scan = lidar.getRangeImage()
    
    pass

# Enter here exit cleanup code.