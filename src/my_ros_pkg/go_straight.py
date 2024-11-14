import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from geometry_msgs.msg import PoseStamped
import math
import numpy as np
import subprocess

if __name__ == "__main__":
    rospy.init_node('mavros_linear', anonymous=True)

    current_state = State()
    pose = PoseStamped()
    target_positions = [(1.0, 0.9), (-0.5, -0.3)]
    reach_tolerance = 0.05  # 오차(m)

    def state_cb(msg):
        global current_state
        current_state = msg

    state_sub = rospy.Subscriber("/mavros/state", State, state_cb)
    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
    arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    pose = PoseStamped()
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 1

    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'
    
    land_set_mode = SetModeRequest()
    land_set_mode.custom_mode = "AUTO.LAND"

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()
    target_index = 0

    while not rospy.is_shutdown():
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            if set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD enabled")
            last_req = rospy.Time.now()
        elif not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            if arming_client.call(arm_cmd).success:
                rospy.loginfo("Vehicle armed")
            last_req = rospy.Time.now()
        else:
            # 목표 지점으로 직선 이동
            target_position = target_positions[target_index]
            pose.pose.position.x = target_position[0]
            pose.pose.position.y = target_position[1]
            pose.pose.position.z = 1  # 고도 유지

            # 현재 위치 계산
            current_position = (pose.pose.position.x, pose.pose.position.y)
            distance_to_target = math.sqrt((current_position[0] - target_position[0])**2 +
                                           (current_position[1] - target_position[1])**2)

            # 목표 지점에 도달했을 때 드론 멈춤 및 detect.py 실행
            if distance_to_target <= reach_tolerance:
                rospy.loginfo(f"Target position {target_position} reached. Hovering and executing detect.py...")

                for _ in range(100):  # 호버링 유지
                    local_pos_pub.publish(pose)
                    rate.sleep()

                # detect.py 실행 및 대기
                detect_process = subprocess.Popen(["python3", "/home/quad/yolov5-python3.6.9-jetson/detect6.py"])
                detect_process.wait()  # detect.py 종료될 때까지 대기

                rospy.loginfo("detect.py completed. Continuing to next target.")

                # 다음 목표로 이동
                target_index += 1
                if target_index >= len(target_positions):
                    rospy.loginfo("All target positions reached. Initiating landing.")
                    set_mode_client.call(land_set_mode)  # 착륙 모드 요청
                    break  # 루프 종료

        local_pos_pub.publish(pose)
        rate.sleep()

