import rospy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CUDA 비활성화
from std_msgs.msg import String
import argparse
import os
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.general import (check_file, check_img_size, non_max_suppression, scale_coords, LOGGER)
from utils.torch_utils import select_device, time_sync
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
import cv2

original_list = {
    "person": 2,
    "TV": 1,
    "desk": 5
}

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
def run(
        weights, source, data, imgsz, conf_thres, iou_thres, max_det, device,
        save_txt, save_conf, save_crop, nosave, classes, agnostic_nms,
        augment, visualize, project, name, exist_ok, line_thickness,
        hide_labels, hide_conf, half, dnn
):
    rospy.loginfo("모델 로딩을 시작합니다...")
    device = select_device(device)
    try:
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        rospy.loginfo("모델 로딩이 완료되었습니다.")
    except Exception as e:
        rospy.logerr(f"모델 로딩 중 오류 발생: {e}")
        return {}

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    rospy.loginfo("모델 설정을 완료했습니다.")

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.startswith(('rtsp://', 'http://', 'https://')) or source.isnumeric()
    if is_file:
        source = check_file(source)

    save_dir = Path(project) / name
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) if webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1 if not webcam else len(dataset)

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    rospy.loginfo("추론을 시작합니다...")
    detected_classes = {}
    detected_frame = None
    history = []  # 최근 프레임의 탐지 결과를 저장할 리스트

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()

        # YOLOv5 전처리
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        t2 = time_sync()
        rospy.loginfo("추론을 수행합니다.")
        pred = model(im, augment=augment)

        t3 = time_sync()
        rospy.loginfo("중복된 바운딩 박스를 제거합니다.")
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        current_classes = {}  # 현재 프레임의 탐지 결과
        if isinstance(im0s, list):  # 웹캠 스트리밍, 다중 프레임 처리
            for i, im0 in enumerate(im0s):
                if len(pred[i]):
                    det = pred[i]
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    detected_frame = im0
                    for *xyxy, conf, cls in det:
                        class_name = names[int(cls)]
                        current_classes[class_name] = current_classes.get(class_name, 0) + 1
        else:  # 단일 이미지 또는 프레임 처리
            if len(pred):
                det = pred[0]
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                detected_frame = im0s
                for *xyxy, conf, cls in det:
                    class_name = names[int(cls)]
                    current_classes[class_name] = current_classes.get(class_name, 0) + 1

        # 탐지 결과 히스토리에 추가
        history.append(current_classes)
        if len(history) > 5:
            history.pop(0)  # 히스토리는 최근 5개만 유지

        # 연속 5개의 프레임에서 동일한 탐지 결과가 있는지 확인
        if len(history) == 5 and all(h == history[0] for h in history):
            rospy.loginfo("5개의 연속된 프레임에서 동일한 탐지 결과를 확인했습니다.")
            break

    # 조건 충족 시 마지막 프레임 저장
    if detected_frame is not None and history[-1]:
        save_path = os.path.expanduser("~/catkin_ws/QUAD/src/yolov5/detected_frame.jpg.jpg")
        cv2.imwrite(save_path, detected_frame)
        rospy.loginfo(f"Saved detected frame to {save_path}")

    return history[-1]  # 마지막 탐지 결과와 프레임 반환

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

def detection_callback(msg):
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

    detected_classes = run(**vars(opt))

    update_list_memory(original_list, detected_classes)

    detection_done = rospy.Publisher('/detection_done', String, queue_size=10)
    detection_done.publish("Detection completed")
    rospy.loginfo("감지 완료 메시지를 발행했습니다.")
    rospy.signal_shutdown("Detection process completed. Shutting down the node.")

def main():
    rospy.init_node('detect.py', anonymous=True)
    rospy.Subscriber('/start_detection', String, detection_callback)
    rospy.loginfo("start_detection 노드가 준비되었습니다. 메시지를 기다리는 중...")
    rospy.spin()

if __name__ == "__main__":
    main()