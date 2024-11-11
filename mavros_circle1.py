import rospy
import subprocess
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool, SetModeRequest, CommandBoolRequest
from geometry_msgs.msg import PoseStamped
import numpy as np

if __name__ == "__main__":
    rospy.init_node('mavros_target', anonymous=True)

    current_state = State()
    pose = PoseStamped()

    def state_cb(msg):
        global current_state
        current_state = msg

    def position_cb(msg):
        global pose
        pose = msg

    # Subscribers and Publishers
    state_sub = rospy.Subscriber("/mavros/state", State, state_cb)
    local_pos_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, position_cb)
    local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

    # Service Clients for arming and mode setting
    arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
    set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

    rate = rospy.Rate(20)

    # 목표 위치 설정
    target_pose = PoseStamped()
    target_pose.pose.position.x = 0.5
    target_pose.pose.position.y = 0.5
    target_pose.pose.position.z = 1.0

    reached_target = False

    while not rospy.is_shutdown():
        # 드론 위치 퍼블리시
        local_pos_pub.publish(target_pose)

        # 목표 좌표와 근사치일 때 (z값 제외)
        if not reached_target and np.isclose(pose.pose.position.x, target_pose.pose.position.x, atol=0.001) and \
           np.isclose(pose.pose.position.y, target_pose.pose.position.y, atol=0.001):

            rospy.loginfo("Target coordinates reached. Stopping the drone.")

            # 목표 위치에 도달한 후 드론의 고도를 유지하기 위해 현재 위치로 설정
            target_pose.pose.position.x = pose.pose.position.x
            target_pose.pose.position.y = pose.pose.position.y
            target_pose.pose.position.z = pose.pose.position.z

            # 파이썬 스크립트 실행
            rospy.loginfo("Executing the script...")
            try:
                process = subprocess.Popen(["python3", "/home/quad/yolov5-python3.6.9-jetson/detect5.py"])
                process.wait()  # 스크립트가 종료될 때까지 기다림
                rospy.loginfo("Script execution finished.")
            except Exception as e:
                rospy.logerr(f"Failed to execute script: {e}")
                break

            # 스크립트가 종료되면 드론 착륙
            rospy.loginfo("Landing the drone.")
            land_set_mode = SetModeRequest()
            land_set_mode.custom_mode = 'AUTO.LAND'

            if set_mode_client.call(land_set_mode).mode_sent:
                rospy.loginfo("Landing initiated.")

            reached_target = True

        rate.sleep()
