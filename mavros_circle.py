############################################################################
# Copyright. QUAD Drone Lab.
# E-Mail. maponarooo@naver.com
# Commercial use or unauthorized copying of this code is prohibited by law.
############################################################################
#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from geometry_msgs.msg import PoseStamped
from math import pow, sqrt
import math
import numpy as np
import subprocess

if __name__ == "__main__":
    rospy.init_node('mavros_circle', anonymous=True)

    current_state = State()
    pose = PoseStamped()
    radius = 0.5

    def state_cb(msg):
        global current_state
        current_state = msg

    state_sub = rospy.Subscriber("/mavros/state", State, state_cb)
    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
    # local_pos_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, position_cb)

    arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(30)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    pose = PoseStamped()

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 1

    # Send a few setpoints before starting
    for i in range(100):
        if(rospy.is_shutdown()):
            break

        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()
    theta = 0.005

    while(not rospy.is_shutdown()):
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()
            else:
                # detect5.py 실행 확인 및 처리
                if detect5_running:
                    # detect5.py가 실행 중일 때 위치 고정
                    local_pos_pub.publish(pose)
                    if detect5_process.poll() is not None:  # detect5.py 종료 여부 확인
                        detect5_running = False
                        rospy.loginfo("detect5.py 종료됨. 드론이 계속 작동 중...")
                else:
                    # "an equation of a circle!"
                    pose.pose.position.x = radius * np.cos(theta)
                    pose.pose.position.y = radius * np.sin(theta)
                    pose.pose.position.z = 1

                    # Check if drone has reached a specific coordinate
                    target_x = 0.5  # Desired x coordinate
                    target_y = 0.0  # Desired y coordinate
                    threshold = 0.001  # Acceptable distance to consider as "reached"

                    distance = sqrt(pow(pose.pose.position.x - target_x, 2) + pow(pose.pose.position.y - target_y, 2))
                    if distance < threshold and not detect5_running:
                        rospy.loginfo(f"Reached target coordinate: ({target_x}, {target_y})")
                        # Execute an external Python script
                        detect5_process = subprocess.Popen(["python3", "/home/quad/yolov5-python3.6.9-jetson/detect5.py"])
                        detect5_running = True

        local_pos_pub.publish(pose)
        #rospy.loginfo(theta)
        rospy.loginfo(f"theta={theta}\nx={pose.pose.position.x}\ny={pose.pose.position.y}\nz={pose.pose.position.z}\n")
       
        theta += 0.005
        if theta > 2*math.pi: # Reset angle to keep it within bounds
            theta = 0.005

        rate.sleep()

