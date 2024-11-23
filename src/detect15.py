import sys
import rospy
import os
from datetime import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
import argparse
import torch
import numpy as np
from utils.general import (non_max_suppression, LOGGER)
from cv_bridge import CvBridge
import cv2
from models.common import DetectMultiBackend

# 전역 변수
colors = {
    'Bottle': (255, 0, 0),  # 빨강
    'Can': (0, 255, 0),  # 초록
    'Ramen': (0, 0, 255),  # 파랑
    'book': (255, 255, 0),  # 노랑
}
zone_id = None  # 구역 정보가 설정되지 않은 상태
history = []  # 감지 히스토리 초기화
detection_completed = False  # 탐지 완료 상태

# 파일 경로 템플릿
DETECTED_FRAME_PATH_TEMPLATE = os.path.expanduser("~/yolov5/zone{zone_id}.jpg")
UPDATED_LIST_PATH_TEMPLATE = os.path.expanduser("~/yolov5/zone{zone_id}_updated_list.txt")

# 리스트 출력 포맷
def print_format(data):
    if isinstance(data, dict):
        return "\n".join(f"{key}: {value}" for key, value in data.items())
    elif isinstance(data, list):
        return "\n".join(f"{item[0]}: {item[1]}" for item in data)
    else:
        return "Invalid data format"

def load_original_list(zone_id):
    file_path = os.path.expanduser(f"~/yolov5/zone{zone_id}.txt")
    if not os.path.exists(file_path):
        rospy.loginfo(f"{file_path} 파일이 존재하지 않습니다. 기본 목록을 사용합니다.")
        return [["Bottle", 2], ["Can", 1], ["Ramen", 5]]  # 기본값
    
    original_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                rospy.logwarn(f"{file_path}에서 잘못된 데이터가 감지되었습니다: '{line}'")
                continue
            try:
                name, count = line.split(":")
                original_list.append([name.strip(), int(count.strip())])
            except ValueError:
                rospy.logwarn(f"{file_path}에서 형식 오류: '{line}'")
                continue
    rospy.loginfo(f"{file_path}에서 zone{zone_id}의 original_list를 불러왔습니다.")
    return original_list

def save_updated_list(zone_id, updated_list):
    file_path = UPDATED_LIST_PATH_TEMPLATE.format(zone_id=zone_id)
    with open(file_path, 'w') as f:
        for item in updated_list:
            f.write(f"{item[0]}: {item[1]}\n")
    rospy.loginfo(f"zone{zone_id}의 최종 업데이트된 original_list를 {file_path}에 저장했습니다.")

def update_list_memory(original_list, detected_list):
    updated = False
    for name, count in detected_list.items():
        found = False
        for item in original_list:
            if item[0] == name:
                found = True
                if item[1] != count:
                    rospy.loginfo(f"--- {name} 수량 업데이트: {item[1]} -> {count}")
                    item[1] = count
                    updated = True
                break
        if not found:
            rospy.loginfo(f"--- {name} 추가.")
            original_list.append([name, count])
            updated = True

    for item in original_list:
        if item[0] not in detected_list and item[1] != 0:
            rospy.loginfo(f"--- {item[0]} 감지되지 않음. 수량 0으로 설정.")
            item[1] = 0
            updated = True

    if updated:
        rospy.loginfo("Updated list:")
        rospy.loginfo(print_format(original_list))
    return original_list

@torch.no_grad()
def load_yolo_model(device, half=False, dnn=False):
    weights = '/home/snow/yolov5/drone2.pt'
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names, pt

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)

    dw //= 2
    dh //= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh
    left, right = dw, dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def process_image(cv_image, model, device):
    img, _, _ = letterbox(cv_image, new_shape=(640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device)
    im = im.float()
    im /= 255.0
    if len(im.shape) == 3:
        im = im[None]

    with torch.amp.autocast('cuda'):
        pred = model(im)

    results = non_max_suppression(pred, 0.7, 0.45, None, False, max_det=1000)
    return results

def display_results(cv_image, results, model):
    detections = results[0]
    if detections is not None:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            color = colors.get(label, (255, 255, 255))

            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(cv_image, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('YOLO Detection', cv_image)
    cv2.waitKey(1)

def zone_callback(msg):
    global zone_id
    new_zone_id = msg.data
    if new_zone_id != zone_id:
        zone_id = new_zone_id
        rospy.loginfo(f"구역이 zone{zone_id}로 변경되었습니다. 탐지 상태를 초기화합니다.")
        reset_detection()
    else:
        rospy.loginfo(f"구역은 이미 zone{zone_id}로 설정되어 있습니다. 초기화하지 않습니다.")

def reset_detection():
    global detection_completed, history
    detection_completed = False
    history = []
    rospy.loginfo("탐지 상태가 초기화되었습니다. 새로운 탐지를 대기합니다.")

def detection_callback(msg, cb_args):
    global zone_id, detection_completed, history
    model, device = cb_args

    if zone_id is None:
        rospy.loginfo("구역 정보가 설정되지 않았습니다. 탐지를 수행하지 않습니다.")
        return

    if detection_completed:
        return

    rospy.loginfo(f"zone{zone_id}에서 감지 명령을 받았습니다. YOLO를 실행합니다...")

    original_list = load_original_list(zone_id)
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    results = process_image(cv_image, model, device)
    display_results(cv_image, results, model)

    # 탐지된 물체 수를 세기 위한 코드
    detected_classes = {}
    if results[0] is not None:
        for det in results[0]:
            label = model.names[int(det[-1])]
            if label in detected_classes:
                detected_classes[label] += 1
            else:
                detected_classes[label] = 1

    if detected_classes:
        history.append(detected_classes)
        if len(history) > 5:
            history.pop(0)

    if len(history) == 5 and all(h == history[0] for h in history):
        rospy.loginfo("5개의 연속된 프레임에서 동일한 탐지 결과를 확인했습니다.")
        
        frame_path = DETECTED_FRAME_PATH_TEMPLATE.format(zone_id=zone_id)
        cv2.imwrite(frame_path, cv_image)
        rospy.loginfo(f"Saved detected frame to {frame_path}")
        
        updated_list = update_list_memory(original_list, detected_classes)
        save_updated_list(zone_id, updated_list)

        detection_done = rospy.Publisher('/detection_done', String, queue_size=10)
        detection_done.publish("Detection completed")
        rospy.loginfo("감지 완료 메시지를 발행했습니다.")

        detection_completed = True
        reset_detection()

def main():
    global history
    rospy.init_node('detect', anonymous=True)

    rospy.Subscriber('/zone_id', Int32, zone_callback)
    rospy.Subscriber('/usb_cam/image_raw', Image, detection_callback, callback_args=(load_yolo_model('cpu')[0], 'cpu'))

    rospy.loginfo("detect.py 노드가 준비되었습니다. 메시지를 기다리는 중...")
    rospy.spin()

if __name__ == "__main__":
    main()