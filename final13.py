# final13.py

import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String
import math

if __name__ == "__main__":
    rospy.init_node('mavros_linear', anonymous=True)

    current_state = State()
    current_pose = PoseStamped()
    pose = PoseStamped()
    target_goal = PoseStamped()
    target_positions_orientations = {
        0: {"position": (0.0, 0.0, 0.0)},
        1: {"position": (-2.46, 2.04, 1.62), "orientation": (0.0, 0.0, -0.69, -0.72)},
        2: {"position": (-2.04, -2.75, 1.2), "orientation": (0.0, 0.0, 0.72, -0.69)},
        3: {"position": (2.05, 3.15, 0.60), "orientation": (0.0, 0.0, -0.99, 0.01)}
    }
    reach_tolerance = 0.1
    detection_done = False  # /detection_done 메시지 수신 여부

    # Detection done subscription initialization flag
    detection_subscribed = False

    def state_cb(msg):
        global current_state
        current_state = msg

    def pose_cb(msg):
        global current_pose
        current_pose = msg

    def detection_done_callback(msg):
        global detection_done
        if msg.data == "Detection completed":  # Check if message matches
            detection_done = True  # Set detection_done flag to True

    # Subscribers and Publishers
    state_sub = rospy.Subscriber("/mavros/state", State, state_cb)
    pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, pose_cb)

    # Subscribe to /detection_done once and print log only on subscription
    if not detection_subscribed:
        rospy.loginfo("Subscribed to /detection_done topic.")
        detection_done_sub = rospy.Subscriber("/detection_done", String, detection_done_callback)
        detection_subscribed = True

    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
    zone_id_pub = rospy.Publisher("/zone_id", Int32, queue_size=10)
    zone_arrived_pub = rospy.Publisher("/zone_arrived", String, queue_size=10)  # 목표 지점 도달 여부 발행

    # MAVROS services
    arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

    rate = rospy.Rate(20)

    rospy.loginfo("Waiting for MAVROS connection...")
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()
    rospy.loginfo("MAVROS connected.")

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 1

    rospy.loginfo("Publishing initial position...")
    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

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

    while not rospy.is_shutdown():
        rospy.loginfo("Waiting for target area number...")
        target_index = input("Enter the target area number (0, 1, 2, 3): ")

        try:
            target_index = int(target_index)
            if target_index not in target_positions_orientations:
                rospy.logwarn("Invalid target area number. Please enter 0, 1, 2, or 3.")
                continue
        except ValueError:
            rospy.logwarn("Invalid input. Please enter a numeric value (0, 1, 2, or 3).")
            continue

        target = target_positions_orientations[target_index]
        target_position = target["position"]
        target_orientation = target.get("orientation", None)

        target_goal.header.stamp = rospy.Time.now()
        target_goal.header.frame_id = "map"
        target_goal.pose.position.x = target_position[0]
        target_goal.pose.position.y = target_position[1]
        target_goal.pose.position.z = current_pose.pose.position.z
        target_goal.pose.orientation = current_pose.pose.orientation

        rospy.loginfo(f"Moving to target position: {target_position}.")
        while not rospy.is_shutdown():
            distance_to_target = math.sqrt(
                (current_pose.pose.position.x - target_position[0]) ** 2 +
                (current_pose.pose.position.y - target_position[1]) ** 2
            )

            if distance_to_target <= reach_tolerance:
                rospy.loginfo(f"Target {target_position} reached. Hovering before adjusting altitude...")

                # Step 1: Hover before adjusting altitude
                hover_start_time = rospy.Time.now()
                while (rospy.Time.now() - hover_start_time).to_sec() < 2.0 and not rospy.is_shutdown():
                    pose.pose.position.x = current_pose.pose.position.x
                    pose.pose.position.y = current_pose.pose.position.y
                    pose.pose.position.z = current_pose.pose.position.z
                    pose.pose.orientation = current_pose.pose.orientation
                    local_pos_pub.publish(pose)  # 현재 위치와 방향 유지
                    rate.sleep()

                # Step 2: Adjust Z (altitude)
                pose.pose.position.z = target_position[2]
                z_adjust_start_time = rospy.Time.now()
                while (rospy.Time.now() - z_adjust_start_time).to_sec() < 3.0 and not rospy.is_shutdown():
                    local_pos_pub.publish(pose)
                    rate.sleep()

                # Step 3: Adjust orientation if specified
                if target_orientation:
                    rospy.loginfo(f"Adjusting orientation to {target_orientation}.")
                    pose.pose.orientation.x = target_orientation[0]
                    pose.pose.orientation.y = target_orientation[1]
                    pose.pose.orientation.z = target_orientation[2]
                    pose.pose.orientation.w = target_orientation[3]
                    orientation_adjust_start_time = rospy.Time.now()
                    while (rospy.Time.now() - orientation_adjust_start_time).to_sec() < 3.0 and not rospy.is_shutdown():
                        local_pos_pub.publish(pose)
                        rate.sleep()

                rospy.loginfo("Target adjustments complete. Publishing zone_arrived and zone_id messages.")
                zone_arrived_pub.publish("Arrived")  # Arrived 메시지 발행
                zone_id_pub.publish(target_index)  # zone_id 발행

                # Wait for user input for next target
                rospy.loginfo("Waiting for user input for next target...")
                while not rospy.is_shutdown() and not detection_done:
                    local_pos_pub.publish(pose)  # Hovering at current position
                    rate.sleep()

                detection_done = False  # Reset detection flag
                break

            local_pos_pub.publish(target_goal)
            rate.sleep()

