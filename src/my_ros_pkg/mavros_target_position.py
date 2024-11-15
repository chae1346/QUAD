# 목표지점으로 이동
# PoseStamped로 현 위치 확인
# 시작과 동시에 yolo 실행. target_position 도달 시 최종 데이터 저리.

import rospy
import subprocess
import threading
import math
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
# from nav_msgs.msg import Odometry # Odometry 메시지를 통해 SLAM이나 위치 추적 정보를 얻을 때 사용
from geometry_msgs.msg import PoseStamped

last_detected = None

def Run_Yolo():
    global last_detected
    command = ["python3", "/home/quad/yolov5-python3.6.9-jetson/detect6.py"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)

    for line in iter(process.stdout.readline, ''):
        last_detected = line.strip() # 감지된 데이터를 저장
        
def main():
    rospy.init_node('mavros_circle', anonymous=True)

    # 초기화
    global current_state, current_position, pose, target_index
    current_state = State()
    current_position = None # SLAM 위치 정보
    pose = PoseStamped()  # MAVROS 위치 정보
    pose.pose.position.z = 1
    rate = rospy.Rate(10)
    target_positions = [(1.0, 0.9), (-0.5, -0.3)] # Target 설정
    reach_tolerance = 0.05  # 오차(m)
    target_index = 0 

    # ROS Topic 처리
    # rospy.Subscriber("/slam/odom", Odometry, odom_callback) # SLAM 위치 정보 수신
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_callback)  # 현위치 정보 수신
    rospy.Subscriber("/mavros/state", State, state_cb) # MAVROS 상태 정보 수신
    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10) # 위치 명령 전송

    arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

    # 연결 대기
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    # MAVROS에 초기 위치 전송
    for i in range(100): 
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    # 오프보드 모드 설정
    offb_set_mode = SetModeRequest(custom_mode='OFFBOARD')
    land_set_mode = SetModeRequest(custom_mode='AUTO.LAND')
    arm_cmd = CommandBoolRequest(value = True)
    last_req = rospy.Time.now()

    # yolo 실행 스레드 시작
    yolov5_thread = threading.Thread(target=Run_Yolo, daemon=True)
    yolov5_thread.start()

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
            # SLAM 데이터 확인 및 목표 위치 설정
            target, distance_to_target = Set_target()

            # 목표 지점에 도달했을 때 호버링 유지 및 데이터 처리
            if target is not None and distance_to_target is not None and distance_to_target <= reach_tolerance:
                rospy.loginfo(f"Target position {target} reached. Hovering...")
                Hovering(local_pos_pub, rate)

                # 다음 목표로 이동
                target_index += 1 
                if target_index >= len(target_positions):
                    rospy.loginfo("All target positions reached. Initiating landing.")
                    if set_mode_client.call(land_set_mode).mode_sent:
                    	rospy.loginfo("Landing mode enabled")
		     else:
		     	rospy.logwarn("Failed to enable landing mode")
                    break  # 루프 종료

        local_pos_pub.publish(pose)
        rate.sleep() # 0.1초 대기

# SLAM 위치 정보를 업데이트
#def odom_callback(msg):
#   global current_position
#   current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y) # data format: Pose(position=Point(x, y, z), orientation=Quaternion(x, y, z, w))

# PoseStamped를 통해 현위치 정보를 업데이트
def pose_callback(msg):
    global current_position
    current_position = (msg.pose.position.x, msg.pose.position.y)

# ROS 드론 현재 상태()
def state_cb(msg):
    global current_state
    current_state = msg

def Hovering(local_pos_pub, rate):
    global pose, last_detected
    for _ in range(100):  # 호버링 유지
        local_pos_pub.publish(pose)
        rate.sleep()
    # yolo 데이터 처리
    if last_detected:
        rospy.loginfo(f"Detected object: {last_detected}")
        rospy.loginfo("detect.py completed. Continuing next target.")
        
# SLAM 데이터 확인 및 목표 위치로 이동 설정, 현 위치와의 거리 계산
def Set_target():
    if current_position is not None and target_index < len(target_positions):
       global target_positions, target_index, pose
       target = target_positions[target_index]
       pose.pose.position.x = target[0]
       pose.pose.position.y = target[1]

       distance_to_target = math.sqrt((current_position[0] - target[0])**2 + (current_position[1] - target[1])**2)
       return target, distance_to_target
    return None, None

if __name__ == "__main__":
    main()
