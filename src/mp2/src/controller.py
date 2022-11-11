import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
from util import euler_to_quaternion, quaternion_to_euler
import math

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp

    def execute(self, currentPose, referencePose):
        # Compute the control input to the vehicle according to the 
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   referencePose: list, the reference state of the vehicle, 
        #       the element in the list are [ref_x, ref_y, ref_theta, ref_v]
        # Output: None
        x_ref = referencePose[0]
        y_ref = referencePose[1]
        theta_ref = referencePose[2]
        v_ref = referencePose[3]
        
        x_b = currentPose.pose.position.x
        y_b = currentPose.pose.position.y
        theta_b = quaternion_to_euler(currentPose.pose.orientation.x,currentPose.pose.orientation.y,currentPose.pose.orientation.z,currentPose.pose.orientation.w)[2]
        v_b = currentPose.twist.linear.x

        # TODO: Implement this function
        delta_x = math.cos(theta_ref)*(x_ref-x_b)+math.sin(theta_ref)*(y_ref-y_b)
        delta_y = -math.sin(theta_ref)*(x_ref-x_b)+math.cos(theta_ref)*(y_ref-y_b)
        delta_theta = theta_ref - theta_b
        delta_v = v_ref - v_b
        delta = np.array([delta_x, delta_y, delta_theta, delta_v])

        #possible K values, pick 1 each for K to tune
        k_x = np.array([0.05, 0.5, 1.0, 0.2])
        k_y = np.array([0.05, 0.1, 0.5, .2, .3])
        k_v = np.array([0.5, 1.0, 1.5, 0.1, 0, 0.2])
        k_theta = np.array([0.8, 1.0, 2.0, 0.2,0,0.5,2.5])
   
        # K = np.array([[k_x[0],0,0,k_v[5]],[0,k_y[2],k_theta[0],0]])
        K = np.array([[k_x[0],0,0,k_v[4]],[0,k_y[2],k_theta[5],0]])
        u = np.dot(K,delta)

        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = u[0]
        newAckermannCmd.steering_angle = u[1]
        # print("ackerman: ", newAckermannCmd.speed, ",", newAckermannCmd.steering_angle)

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)


    def setModelState(self, currState, targetState, vehicle_state = "run"):
        control = self.rearWheelFeedback(currState, targetState)
        self.controlPub.publish(control)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)