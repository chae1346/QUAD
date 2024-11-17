"""
드론 (Jetson Nano)
 ├── ROS 노드: SLAM (RealSense)
 ├── ROS 노드: MAVROS (OFFBOARD 제어)
 ├── 카메라 스트리밍 서버 (GStreamer, OpenCV 등)
     └─> 노트북 (YOLO 실행)
          └─> 결과를 ROS 토픽으로 Jetson Nano에 전달
"""

"""
1. 오프보드 코드가 목표 지점 도달 여부를 판단:
    SLAM으로부터 토픽을 구독하지 못했을 경우, MAVROS 데이터 및 키보드 인터럽트 사용.
    목표 지점으로 이동 및 도달 여부 확인.

2. 목표 지점 도달 시:
    호버링 유지.
    YOLO 실행 토픽 발행.

3. 노트북에서 객체 탐지 실행:
    객체 탐지 완료 알림 토픽 발행.

4. Jetson Nano가 객체 탐지 결과를 구독:
    후속 행동(예: 특정 객체로 이동, 다음 목표로 전환)을 수행.
"""


import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from nav_msgs.msg import Odometry # SLAM에서 현 위치 정보를 얻을 때 사용
from geometry_msgs.msg import Point, PoseStamped # SLAM에서 목표좌표 수신, MAVROS와 드론 간 메시지 송수신
from std_msgs.msg import String
import math

def main():
    rospy.init_node('drone_offboard', anonymous=True)

    # 초기화
    global current_state, current_position, slam_position, mavros_position, pose, target_index
    current_state = State()
    current_position = None
    slam_position = None
    mavros_position = None
    pose = PoseStamped()  # MAVROS 위치 정보
    pose.pose.position.z = 1
    rate = rospy.Rate(10)
    target_positions = [] # Target list
    target_index = 0
    reach_tolerance = 0.05  # 오차(m)

    # ROS topic 처리
    rospy.Subscriber("/slam/odom", Odometry, odom_callback) # SLAM 위치 정보 수신
    rospy.Subscriber("/mavros/state", State, state_callback) # MAVROS 상태 정보 수신
    rospy.Subscriber("/slam/target_positions", Point, target_callback)  # SLAM 목표 좌표 수신

    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10) # MAVROS에 목표위치 및 방향 발행
    detection_pub = rospy.Publisher("/start_detection", String, queue_size=10)  # 노트북에 yolo 실행 명령 발행
    rospy.sleep(1)

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

    # OFFBOARD 모드 설정
    offb_set_mode = SetModeRequest(custom_mode='OFFBOARD')
    land_set_mode = SetModeRequest(custom_mode='AUTO.LAND')
    arm_cmd = CommandBoolRequest(value = True)
    last_req = rospy.Time.now()


    while not rospy.is_shutdown():
        # OFFBOARD 모드 전환
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
            set_mode_client.call(offb_set_mode)
            rospy.loginfo("OFFBOARD mode enabled.")
            last_req = rospy.Time.now()

        if not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(1.0):
            arming_client.call(arm_cmd)
            rospy.loginfo("Vehicle armed.")
            last_req = rospy.Time.now()


        # 현 위치 업데이트
        if slam_position is not None:
            current_position = slam_position
        else:
            rospy.logwarn("SLAM not available. Using MAVROS position as fallback.")
            current_position = mavros_position
        

        # Target 토픽 확인
        if not target_positions:
            rospy.logwarn("Target positions are empty.")
            rospy.loginfo("Input target x, y.")
            keyboard_target = get_keyboard_input()
            if keyboard_target:
                target_positions.append(keyboard_target)
            rate.sleep()
            continue
        
        # 목표 위치로 이동
        target, distance_to_target, pose = Set_target()

        # 목표 지점에 도달했을 때 호버링 유지
        if target is not None and distance_to_target is not None and distance_to_target <= reach_tolerance:
            rospy.loginfo(f"Target: {target}. Reached at {current_position}.")
            Hovering(local_pos_pub, rate)

            # ROS 메시지를 통해 노트북에 detect.py 실행 명령 전송
            detection_pub.publish("start_detection")
            rospy.loginfo("Detection command sent via ROS.")
            
            rospy.Subscriber("/detection_done", String, detection_done_callback)  # 탐지 완료 토픽 구독
            rospy.loginfo("Waiting for detection_done messages...")
            rospy.spin()

            # 다음 목표로 이동
            rospy.loginfo("Moving to next target.")                
            target_index += 1
            
            if target_index >= len(target_positions):
                rospy.loginfo("All target positions reached. Initiating landing.")
                if set_mode_client.call(land_set_mode).mode_sent:
                    rospy.loginfo("Landing mode enabled.")
                else:
                    rospy.logwarn("Failed to enable landing mode.")
                break  # 루프 종료

        local_pos_pub.publish(pose)
        rate.sleep() # 0.1초 대기



def odom_callback(msg):
    """ SLAM 위치 정보 업데이트 """
    global slam_position
    slam_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)


def pose_callback(msg):
    """ MAVROS 위치 정보 업데이트 """
    global mavros_position
    mavros_position = (msg.pose.position.x, msg.pose.position.y)


def state_callback(msg):
    """ MAVROS 드론 현재 상태 업데이트 """
    global current_state
    current_state = msg


def target_callback(msg):
    """ SLAM 목표 좌표를 수신하여 리스트에 추가 """
    global target_positions
    target_positions.append((msg.x, msg.y))  # SLAM에서 받은 목표 좌표 추가
    rospy.loginfo(f"Received new target position: ({msg.x}, {msg.y})")


def detection_done_callback(msg):
    """ 탐지 완료 여부 확인 """
    rospy.loginfo(f"Detection done.")


def Set_target():
    """ 목표 위치로 이동, 현 위치와의 거리 계산 """
    global target_positions, target_index

    target = target_positions[target_index]
    pose.pose.position.x = target[0]
    pose.pose.position.y = target[1]

    distance_to_target = math.sqrt((current_position[0] - target[0])**2 + (current_position[1] - target[1])**2)
    return target, distance_to_target


def Hovering(local_pos_pub, rate, duration=10):
    global pose
    rospy.loginfo("Hovering at current position...")
    start_time = rospy.Time.now()

    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < duration:
        local_pos_pub.publish(pose)
        rate.sleep()

    rospy.loginfo("Hovering completed.")


def get_keyboard_input():
    x = float(input("Enter target X coordinate: "))
    y = float(input("Enter target Y coordinate: "))
    return (x, y)



if __name__ == "__main__":
    main()
