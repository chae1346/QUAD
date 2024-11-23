import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

def send_goal_coordinates(zone_number):
    coordinates = {
        1: (-1.3, -1.2),
        2: (3.0, 4.0),
        3: (5.0, -6.0)
    }
    
    if zone_number not in coordinates:
        rospy.logerr("Invalid zone number. Please enter 1, 2, or 3.")
        return

    x, y = coordinates[zone_number]
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 1.0  # Set altitude to 1 meter
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0
    goal.pose.orientation.w = 1.0
    
    rospy.loginfo(f"Sending goal to coordinates: ({x}, {y}, altitude: 1.0)")
    goal_pub.publish(goal)
    print(f"[INFO] Goal sent to coordinates: ({x}, {y}, altitude: 1.0)")

def path_callback(path):
    rospy.loginfo("Received path from path topic.")
    print("[INFO] Received path from path topic.")
    waypoints = []
    if len(path.poses) == 0:
        rospy.logwarn("No poses received in the path.")
        print("[WARN] No poses received in the path.")
        return

    for i in range(len(path.poses) - 1):
        start = path.poses[i].pose.position
        end = path.poses[i + 1].pose.position

        distance = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
        num_waypoints = int(distance / 0.3)

        for j in range(num_waypoints):
            new_x = start.x + (end.x - start.x) * (j / num_waypoints)
            new_y = start.y + (end.y - start.y) * (j / num_waypoints)
            waypoints.append((new_x, new_y))
    
    rospy.loginfo("Generated waypoints:")
    for waypoint in waypoints:
        rospy.loginfo(f"Waypoint: {waypoint}")
    
    # Display generated waypoints in terminal
    print("Generated waypoints:")
    for waypoint in waypoints:
        print(f"Waypoint: {waypoint}")

def dynamic_path_subscriber():
    path_topic = "/move_base/NavfnROS/plan"
    rospy.Subscriber(path_topic, Path, path_callback)
    rospy.loginfo(f"Subscribed to path topic: {path_topic}")
    print(f"[INFO] Subscribed to path topic: {path_topic}")
    print("[INFO] Waiting for path topic messages...")

if __name__ == "__main__":
    rospy.init_node('waypoint_generator')
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    
    try:
        zone_number = int(input("Enter zone number (1, 2, or 3): "))
        print(f"[INFO] Sending goal for zone number: {zone_number}")
        send_goal_coordinates(zone_number)
        dynamic_path_subscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


