#!/usr/bin/env python
import rospy
import numpy as np
from math import cos, sin
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3

# Publisher for sending acceleration commands to flappy bird
pub_acc_cmd = rospy.Publisher('/flappy_acc', Vector3, queue_size=1)
# Setting Numpy print options
np.set_printoptions(precision=2, suppress=True)

dt = 1/30.  # time step
v_max = 0.5  # maximum velocity
a_max = np.array([3., 30.], dtype=np.float32)  # maximum acceleration allowed
min_clearance = 1.5  # minimum clearance
DEBUG = False


def initNode():
    # Here we initialize our node running the automation code
    rospy.init_node('flappy_automation_code', anonymous=True)

    # Subscribe to topics for velocity and laser scan from Flappy Bird game
    rospy.Subscriber("/flappy_vel", Vector3, velCallback)
    rospy.Subscriber("/flappy_laser_scan", LaserScan, laserScanCallback)

    # Initialize global variables
    global flappy_v, v_desired, last_ranges

    flappy_v = np.zeros((2,))  # flappy current velocity
    v_desired = np.zeros((2,))  # flappy desired velocity
    last_ranges = np.array([], dtype=np.float32).reshape(0, 9)  # memory of last 14 laser ranges

    # Ros spin to prevent program from exiting
    rospy.spin()


def velCallback(msg):
    # msg has the format of geometry_msgs::Vector3
    # Example of publishing acceleration command on velocity velCallback

    global flappy_v

    # Update current velocity with exact sensor reading
    flappy_v = np.array([msg.x, msg.y], dtype=np.float32)

    # Compute desired acceleration in allowed range
    a_desired = (v_desired - flappy_v) / dt
    a_desired = np.maximum(a_desired, - a_max)
    a_desired = np.minimum(a_desired, a_max)

    # Publish new acceleration command as Vector3
    pub_acc_cmd.publish(Vector3(a_desired[0], a_desired[1], 0.))
    print('\nv* = {}'.format(v_desired))
    print('v = {}, a = {}'.format(flappy_v, a_desired))


def laserScanCallback(msg):
    # msg has the format of sensor_msgs::LaserScan
    # print laser angle and range

    global v_desired, last_ranges

    # Convert new laser ranges to numpy array
    new_ranges = np.array(msg.ranges, dtype=np.float32)

    if DEBUG:
        print('\nnew ranges = {}'.format(new_ranges))

    # Erase the least recent laser range
    if last_ranges.shape[0] > 14:
        last_ranges = np.delete(last_ranges, 0, 0)

    last_ranges = np.vstack((last_ranges, new_ranges))  # Stack new laser ranges to heap
    laser_ranges = np.min(last_ranges, axis=0)  # Process last 15 laser ranges to avoid spurious peaks in readings
    laser_angles = np.arange(msg.angle_min, 1.01 * msg.angle_max, msg.angle_increment, dtype=np.float32)  # Laser angles

    if DEBUG:
        print('laser ranges = {}'.format(laser_ranges))
        print('laser angles = {}'.format(laser_angles))

    n = len(laser_ranges) - 1  # number of laser ranges

    # If straight we have enough free space
    if laser_ranges[n//2] > min_clearance:
        v_desired = np.array([v_max, 0], dtype=np.float32)  # then go straight

        if DEBUG:
            print('Going straight')

    else:  # else move in the direction of the best peak detected, corresponding to the clearest direction
        peaks_clearance, peaks_angles = [], []

        # Check if first laser range is a peak
        if laser_ranges[0] >= laser_ranges[1]:
            peaks_angles.append(laser_angles[0])
            # Approximate the clearance (peak range + side clearance)
            peaks_clearance.append(laser_ranges[0] + 1.6 * laser_ranges[1])

        # Find peaks which are not going straight
        for i in range(1, n):
            if i != n//2 and laser_ranges[i] >= laser_ranges[i-1] and laser_ranges[i] >= laser_ranges[i+1]:
                peaks_angles.append(laser_angles[i])
                # Approximate the clearance (peak range + side clearance)
                peaks_clearance.append(laser_ranges[i] + 0.8 * (laser_ranges[i+1] + laser_ranges[i-1]))

        # Check if last laser range is a peak
        if laser_ranges[n] >= laser_ranges[n-1]:
            peaks_angles.append(laser_angles[n])
            # Approximate the clearance (peak range + side clearance)
            peaks_clearance.append(laser_ranges[n] + 1.6 * laser_ranges[n-1])

        # Convert to Numpy array
        peaks_angles = np.array(peaks_angles, dtype=np.float32)
        peaks_clearance = np.array(peaks_clearance, dtype=np.float32)

        if DEBUG:
            print('peaks angles = {}'.format(peaks_angles))
            print('peaks integrals = {}'.format(peaks_clearance))

        # Compute a score to maximize based on clearance and smoothness
        dv_desired = v_max * np.vstack((np.cos(peaks_angles), np.sin(peaks_angles))).T - v_desired
        smoothness = np.sqrt(dv_desired[:, 0] ** 2 + dv_desired[:, 1] ** 2)
        scores = peaks_clearance - smoothness

        # Compute best direction and desired velocity based on score
        best_direction = peaks_angles[np.argmax(scores)]
        v_desired = v_max * np.array([cos(best_direction), sin(best_direction)], dtype=np.float32)

        if DEBUG:
            print('scores = {}'.format(scores))
            print('best_angle = {:.3f}'.format(best_direction))

    # Whatever the selected direction of movement is, close obstacles exert a weak repulsive force
    close_obstacles = laser_ranges < 1.
    v_repulsive = - 0.03 / laser_ranges[close_obstacles] ** 2
    v_repulsive_x = np.sum(np.dot(v_repulsive, np.cos(laser_angles[close_obstacles])))
    v_repulsive_y = np.sum(np.dot(v_repulsive, np.sin(laser_angles[close_obstacles])))
    v_desired += np.array([0.2 * v_repulsive_x, v_repulsive_y], dtype=np.float32)

    # Whatever the selected direction of movement is, far obstacles exert a strong repulsive force
    far_obstacles = (laser_ranges >= 1.) & (laser_ranges < 2.)
    v_repulsive = - 0.20 / laser_ranges[far_obstacles] ** 2
    v_repulsive_x = np.sum(np.dot(v_repulsive, np.cos(laser_angles[far_obstacles])))
    v_repulsive_y = np.sum(np.dot(v_repulsive, np.sin(laser_angles[far_obstacles])))
    v_desired += np.array([0.2 * v_repulsive_x, v_repulsive_y], dtype=np.float32)
    

if __name__ == '__main__':
    try:
        initNode()
    except rospy.ROSInterruptException:
        pass
