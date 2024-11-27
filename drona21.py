# drona21.py

import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math

if __name__ == "__main__":
    rospy.init_node('mavros_linear', anonymous=True)

    current_state = State()
    current_pose = PoseStamped()
    pose = PoseStamped()
    target_goal = PoseStamped()
    target_positions_orientations = [
        {"position": (-2.14, 0.19, 1.5), "orientation": (0.0, 0.0, -1.0, 0.0)},
        {"position": (-2.09, -1.70, 0.7), "orientation": (0.0, 0.0, 0.74, -0.66)},
        {"position": (-2.04, 1.58, 1.2), "orientation": (0.0, 0.0, -0.67, -0.74)}
    ]
    reach_tolerance = 0.1  # 목표 도달 거리 오차
    hover_after_orientation_duration = 5.0  # 방향 변경 후 추가 호버링 시간 (초)

    def state_cb(msg):
        global current_state
        current_state = msg

    def pose_cb(msg):
        global current_pose
        current_pose = msg

    def path_cb(msg):
        global current_path
        current_path = msg

    current_path = Path()
    state_sub = rospy.Subscriber("/mavros/state", State, state_cb)
    pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)
    path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, path_cb)
    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

    rate = rospy.Rate(20)  # 20Hz 퍼블리시

    rospy.loginfo("Waiting for MAVROS connection...")
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()
    rospy.loginfo("MAVROS connected.")

    # 초기 목표 위치 설정
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 1

    # 초기 목표 위치 퍼블리시
    rospy.loginfo("Publishing initial position...")
    for i in range(100):  # OFFBOARD 모드 전환을 위한 초기 위치 퍼블리시
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    # OFFBOARD 모드 전환 및 아밍
    rospy.loginfo("Switching to OFFBOARD mode and arming...")
    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    while not rospy.is_shutdown() and (current_state.mode != "OFFBOARD" or not current_state.armed):
        if current_state.mode != "OFFBOARD":
            rospy.loginfo("Trying to enable OFFBOARD mode...")
            if set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD mode enabled.")
        if not current_state.armed:
            rospy.loginfo("Trying to arm the vehicle...")
            if arming_client.call(arm_cmd).success:
                rospy.loginfo("Vehicle armed.")
        rate.sleep()

    rospy.loginfo("OFFBOARD mode and arming complete. Ready for flight.")

    # 메인 루프 시작
    target_index = 0
    while not rospy.is_shutdown():
        if target_index < len(target_positions_orientations):
            target = target_positions_orientations[target_index]
            target_position = target["position"]
            target_orientation = target["orientation"]

            # 목표 지점 설정 (현재 고도와 오리엔테이션 유지)
            target_goal.header.stamp = rospy.Time.now()
            target_goal.header.frame_id = "map"
            target_goal.pose.position.x = target_position[0]
            target_goal.pose.position.y = target_position[1]
            target_goal.pose.position.z = current_pose.pose.position.z  # 이동 중 현재 고도 유지
            target_goal.pose.orientation = current_pose.pose.orientation  # 이동 중 현재 오리엔테이션 유지

            rospy.loginfo(f"Publishing goal to move_base: {target_position} with current orientation.")
            goal_pub.publish(target_goal)
            rospy.sleep(1)

            if current_path.poses:
                rospy.loginfo("Following the planned path.")
                for pose_stamped in current_path.poses:
                    pose = pose_stamped
                    pose.pose.position.z = current_pose.pose.position.z  # 이동 중 현재 고도 유지
                    pose.pose.orientation = current_pose.pose.orientation  # 이동 중 현재 오리엔테이션 유지
                    local_pos_pub.publish(pose)
                    rate.sleep()

                    # 목표 위치에 도달했는지 확인
                    distance_to_target = math.sqrt(
                        (current_pose.pose.position.x - target_position[0]) ** 2 +
                        (current_pose.pose.position.y - target_position[1]) ** 2 +
                        (current_pose.pose.position.z - current_pose.pose.position.z) ** 2
                    )

                    if distance_to_target <= reach_tolerance:
                        rospy.loginfo(f"Target position {target_position} reached. Adjusting z-axis and hovering...")
                        
                        # 목표 고도로 변경하고 3초간 호버링
                        hover_start_time = rospy.Time.now()
                        while (rospy.Time.now() - hover_start_time).to_sec() < 3.0:
                            pose.pose.position.x = target_position[0]
                            pose.pose.position.y = target_position[1]
                            pose.pose.position.z = target_position[2]  # 목표 z값으로 변경
                            pose.pose.orientation = current_pose.pose.orientation  # 현재 오리엔테이션 유지
                            local_pos_pub.publish(pose)
                            rate.sleep()

                        rospy.loginfo("Aligning to target orientation.")
                        pose.pose.orientation.x = target_orientation[0]
                        pose.pose.orientation.y = target_orientation[1]
                        pose.pose.orientation.z = target_orientation[2]
                        pose.pose.orientation.w = target_orientation[3]

                        # 목표 오리엔테이션으로 전환하고 5초간 호버링
                        for _ in range(50):  # 5초 동안 10Hz로 퍼블리시
                            local_pos_pub.publish(pose)
                            rate.sleep()

                        rospy.loginfo("Hovering after orientation change for 5 seconds...")
                        hover_after_orientation_start = rospy.Time.now()
                        while (rospy.Time.now() - hover_after_orientation_start).to_sec() < 5.0:
                            local_pos_pub.publish(pose)
                            rate.sleep()

                        rospy.loginfo("Hovering complete. Moving to the next target.")
                        target_index += 1
                        break
            else:
                rospy.logwarn("No path received from move_base. Retrying...")
                rospy.sleep(2)

        if target_index >= len(target_positions_orientations):
            rospy.loginfo("All target positions reached. Initiating landing.")
            land_set_mode = SetModeRequest()
            land_set_mode.custom_mode = "AUTO.LAND"
            set_mode_client.call(land_set_mode)
            break

        rate.sleep()

