import rospy
import numpy as np
import argparse

from gazebo_msgs.msg import  ModelState
from controller.controller import VehicleController
from perception.perception import VehiclePerception
from decision.decision import VehicleDecision
import time
from util.util import euler_to_quaternion, quaternion_to_euler
import pickle

def run_model(model_name):
    rospy.init_node("gem1_dynamics")
    rate = rospy.Rate(100)  # 100 Hz

    perceptionModule = VehiclePerception(model_name)
    decisionModule = VehicleDecision()
    controlModule = VehicleController(model_name)
    while not rospy.is_shutdown():
        # res = sensors.lidarReading()
        # print(res)
        rate.sleep()  # Wait a while before trying to get a new state

        # Get the current position and orientation of the vehicle
        currState =  perceptionModule.gpsReading()
        front_dist = perceptionModule.lidarReading()
        lateral_error, lane_theta = perceptionModule.cameraReading()

        target_v, lateral_error, lane_theta = decisionModule.get_ref_state(currState, front_dist, lateral_error, lane_theta)

        controlModule.execute(target_v, lateral_error, lane_theta)

if __name__ == "__main__":
    run_model('gem')
