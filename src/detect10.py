import sys
sys.path.remove('/home/cherry/catkin_ws/QUAD/src/yolov5')

import rospy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CUDA 비활성화
from sensor_msgs.msg import Image
from std_msgs.msg import String
import argparse
import os
from pathlib import Path
import torch
import numpy as np
from utils.general import (check_file, check_img_size, non_max_suppression, scale_coords, LOGGER)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from cv_bridge import CvBridge
import cv2
from models.common import DetectMultiBackend

original_list = {
    "person": 2,
    "TV": 1,
    "desk": 5
}

colors = {
    'person': (255, 0, 0),  # 빨강
    'TV': (0, 255, 0),  # 초록
    'chair': (0, 0, 255),  # 파랑
    'book': (255, 255, 0),  # 노랑
    # 필요에 따라 다른 클래스 추가
}

history = []

def print_format(data):
    return "\n".join(f"{key}: {value}" for key, value in data.items())

def update_list_memory(original_list, detected_list):
    print("Original list:")
    print(print_format(original_list))

    print("\nDetected list:")
    print(print_format(detected_list))
    print(f"\n")

    for name, count in detected_list.items():
        if name not in original_list:
            print(f"--- {name} 추가.")
            original_list[name] = count
        elif original_list[name] != count:
            print(f"--- {name} 수량 업데이트.")
            original_list[name] = count
        else:
            print(f"--- {name} 수량 일치.")
    
    for name in list(original_list.keys()):  # 키 목록 복사하여 안전하게 반복
        if name not in detected_list:
            print(f"--- {name} 감지되지 않음.")
            original_list[name] = 0

    print("\nUpdated list:")
    print(print_format(original_list))
    print("\n")

    return original_list

@torch.no_grad()
def load_yolo_model(device, half=False, dnn=False):
    weights = '/home/cherry/catkin_ws/QUAD/src/yolov5/yolov5s.pt'
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt = model.stride, model.names, model.pt  # pt 속성 추가
    return model, stride, names, pt

def process_image(cv_image, model, device):
    # YOLOv5 모델을 통해 이미지 처리
    img = cv2.resize(cv_image, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device)
    im = im.float()  # uint8 to fp16/32
    im /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    with torch.amp.autocast('cuda'):
        pred = model(im)

    # NMS (중복된 바운딩 박스 제거)
    results = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    return results

def display_results(cv_image, results, model):
    # YOLOv5 모델의 결과를 이미지에 표시
    detections = results[0]
    if detections is not None:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            label = model.names[int(cls)]
            color = colors.get(label, (255, 255, 255))  # 클래스에 따라 색상 설정

            # 바운딩 박스와 텍스트 그리기
            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(cv_image, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 이미지 표시
    cv2.imshow('YOLO Detection', cv_image)
    cv2.waitKey(1)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='home/catkin_ws/QUAD/src/yolov5/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels on bounding boxes')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences on bounding boxes')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args([])
    return opt

def detection_callback(msg, cb_args):
    model, device = cb_args
    rospy.loginfo("감지 명령을 받았습니다. YOLO를 실행합니다...")

    opt = parse_opt()
    opt.source = 0
    opt.imgsz = [320, 320]  # 이미지 크기
    opt.conf_thres = 0.25  # Confidence threshold
    opt.iou_thres = 0.45  # IoU threshold for NMS
    opt.device = 'cpu'  # GPU 사용
    opt.classes = None  # 모든 클래스 감지
    opt.augment = False
    opt.hide_labels = False
    opt.hide_conf = False
    opt.half = False
    opt.dnn = False
    
    bridge = CvBridge()
    # ROS 이미지를 OpenCV 이미지로 변환
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    # YOLOv5로 이미지 처리 및 표시
    results = process_image(cv_image, model, device)
    display_results(cv_image, results, model)

    # 탐지된 객체 정보를 추출하여 업데이트
    detected_classes = {model.names[int(det[-1])]: 1 for det in results[0]}

    # 탐지된 객체 정보가 있을 때만 history에 추가
    if detected_classes:
        history.append(detected_classes)
        if len(history) > 5:
            history.pop(0)  # 히스토리는 최근 5개만 유지
    else:
        rospy.loginfo("No objects detected; history not updated.")

    # 연속 5개의 프레임에서 동일한 탐지 결과가 있는지 확인
    if len(history) == 5 and all(h == history[0] for h in history):
        rospy.loginfo("5개의 연속된 프레임에서 동일한 탐지 결과를 확인했습니다.")
        # 조건이 충족될 때 마지막 프레임을 저장
        save_path = os.path.expanduser("~/catkin_ws/QUAD/src/yolov5/detected_frame.jpg")
        cv2.imwrite(save_path, cv_image)
        rospy.loginfo(f"Saved detected frame to {save_path}")
        
        update_list_memory(original_list, detected_classes)

        detection_done = rospy.Publisher('/detection_done', String, queue_size=10)
        detection_done.publish("Detection completed")
        rospy.loginfo("감지 완료 메시지를 발행했습니다.")
        rospy.signal_shutdown("프로그램을 종료합니다.")

def main():
    rospy.init_node('detect', anonymous=True)

    # 옵션 설정 및 모델 로드
    opt = parse_opt()
    model, stride, names, pt = load_yolo_model(opt.device, half=opt.half, dnn=opt.dnn)

    # ROS 이미지 토픽 구독 및 콜백 함수 설정
    rospy.Subscriber('/usb_cam/image_raw', Image, detection_callback, callback_args=(model, opt.device))
    rospy.loginfo("detect.py 노드가 준비되었습니다. 메시지를 기다리는 중...")

    rospy.spin()

if __name__ == "__main__":
    main()