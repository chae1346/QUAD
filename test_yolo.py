import rospy
from std_msgs.msg import String

def start_detection():
    rospy.init_node('start_detection_node', anonymous=True)

    # Publisher to send start detection command
    detection_pub = rospy.Publisher('/start_detection', String, queue_size=10)

    # Subscriber to listen for completion message
    def detection_done_callback(msg):
        rospy.loginfo(f"Detection Done Message Received: {msg.data}")
        rospy.signal_shutdown("Detection completed.")

    rospy.Subscriber('/detection_done', String, detection_done_callback)

    rospy.loginfo("Publishing start detection command...")
    detection_pub.publish("Start Detection")
    rospy.loginfo("Waiting for detection to complete...")

    rospy.spin()  # Wait for the detection to complete

if __name__ == "__main__":
    start_detection()

