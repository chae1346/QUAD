#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String  # 새로운 토픽에 사용할 메시지 타입
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from pathlib import Path

class YoloRealsenseNode:
    def __init__(self):
        rospy.init_node('yolo_realsense_node', anonymous=True)
        self.bridge = CvBridge()
        self.model = self.load_yolo_model()
        
        # Image topic을 구독
        rospy.Subscriber('usb_cam/image_raw', Image, self.image_callback)
        
        # 새로운 토픽을 생성 (객체 타입과 개수를 보내기 위한 토픽)
        self.detection_pub = rospy.Publisher('/yolo/detections', String, queue_size=10)

        # 클래스별 색상 설정
        self.colors = {
            'bottle': (255, 0, 0),  # 빨강
            'refrigerator': (0, 255, 0),  # 초록
            'chair': (0, 0, 255),  # 파랑
            'book': (255, 255, 0),  # 노랑
            # 필요에 따라 다른 클래스 추가
        }

    def load_yolo_model(self):
        # YOLO 모델 로드
        model_path = Path("/home/snow/yolo_realsense_2/scripts/yolov5n.pt")  # 모델 파일 경로
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        return model

    def image_callback(self, msg):
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # YOLOv5로 이미지 처리
            results = self.process_image(cv_image)
            # 결과를 화면에 표시
            self.display_results(cv_image, results)
            # 객체 정보 게시
            self.publish_detections(results)
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

    def process_image(self, cv_image):
        # YOLOv5 모델을 통해 이미지 처리
        with torch.amp.autocast('cuda'):
            results = self.model(cv_image)
        return results

    def display_results(self, cv_image, results):
        # YOLOv5 모델의 결과를 이미지에 표시
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            # 사각형 그리기
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']

            # 클래스에 해당하는 색상 가져오기 (기본값은 흰색)
            color = self.colors.get(label, (255, 255, 255))  # 클래스에 따라 색상 설정

            # 바운딩 박스와 텍스트 그리기
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 이미지 표시
        cv2.imshow('YOLO Detection', cv_image)
        cv2.waitKey(1)  # 화면 업데이트를 위해 잠시 대기

    def publish_detections(self, results):
        # 인식된 객체의 타입과 개수를 문자열로 변환하여 새로운 토픽에 게시
        detections = results.pandas().xyxy[0]
        labels = detections['name'].value_counts()  # 각 객체 타입별 개수 계산
        detection_info = []

        # 각 객체 타입과 개수 추출
        for label, count in labels.items():
            detection_info.append(f"{label}: {count}")

        # 최종적으로 객체 타입과 개수를 하나의 문자열로 병합
        detection_message = ', '.join(detection_info)

        # 메시지를 토픽에 게시
        rospy.loginfo(f"Publishing detection info: {detection_message}")
        self.detection_pub.publish(detection_message)

if __name__ == '__main__':
    try:
        yolo_node = YoloRealsenseNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
