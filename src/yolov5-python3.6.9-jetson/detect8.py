import rospy
from std_msgs.msg import String
import argparse
import os
from pathlib import Path
from collections import Counter
import datetime
import torch
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import check_file, check_img_size, non_max_suppression, scale_coords, LOGGER
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors
import cv2


@torch.no_grad()
def run(
        weights, source, data, imgsz, conf_thres, iou_thres, max_det, device,
        save_txt, save_conf, save_crop, nosave, classes, agnostic_nms,
        augment, visualize, project, name, exist_ok, line_thickness,
        hide_labels, hide_conf, half, dnn
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    webcam = source.startswith(('rtsp://', 'http://', 'https://')) or source.isnumeric()
    if is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) if webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1 if not webcam else len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    detected_classes = []
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()

        pred = model(im, augment=augment)
        t3 = time_sync()

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                detected_classes += [names[int(cls)] for *xyxy, conf, cls in det]
    return detected_classes


def save_detected_classes_to_txt(detected_classes, output_path='detected_classes.txt'):
    class_counts = Counter(detected_classes)
    with open(output_path, 'w') as f:
        for class_name, count in class_counts.items():
            f.write(f"{class_name}: {count}\n")


def read_file_content(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def compare_and_update(detected_classes, list_file='List.txt', detected_file='detected_classes.txt', last_image=None, save_path='final_detected_image.jpg'):
    if not detected_classes:
        print("No detected classes to compare. Skipping...")
        return
    save_detected_classes_to_txt(detected_classes, detected_file)
    list_content = read_file_content(list_file)
    detected_content = read_file_content(detected_file)

    if list_content == detected_content:
        print("Contents are the same. Saving the last detected image and exiting...")
        if last_image is not None:
            cv2.imwrite(save_path, last_image)
            print(f"Last detected image saved to {save_path}")

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(detected_file, 'a', encoding='utf-8') as f:
            f.write(f"\nTimestamp: {current_time}\n")
        return
    else:
        print("Contents are different. Continuing...")


def detection_callback(msg):
    rospy.loginfo("Received detection command. Running YOLO on live stream...")
    
    # YOLO 실행 옵션 설정
    opt = parse_opt()
    opt.source = 'rtsp://example.com/live_stream'  # 카메라 스트리밍 URL
    opt.imgsz = [640, 640]  # 이미지 크기
    opt.conf_thres = 0.25  # Confidence threshold
    opt.iou_thres = 0.45  # IoU threshold for NMS
    opt.device = '0'  # GPU 사용
    opt.classes = None  # 모든 클래스 감지
    opt.augment = False  # 증강 비활성화

    detected_classes = run(**vars(opt)) or []
    
    if not detected_classes:
        rospy.loginfo("No detected classes. Continuing...")
    else:
        rospy.loginfo(f"Detected classes: {detected_classes}")
    
    # 결과 비교 및 업데이트
    compare_and_update(detected_classes)
    rospy.loginfo("YOLO detection on live stream completed.")

    # 객체 인식 완료 신호 발행
    detection_done_pub.publish("Detection completed")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args([])
    return opt


def main():
    global detection_done_pub

    rospy.init_node('yolo_detection_node', anonymous=True)

    # 객체 인식 완료 토픽 퍼블리셔 설정
    detection_done_pub = rospy.Publisher('/detection_done', String, queue_size=10)

    # 객체 인식 실행 명령 토픽 구독
    rospy.Subscriber('/start_detection', String, detection_callback)
    
    rospy.loginfo("YOLO detection node is ready and waiting for commands...")
    rospy.spin()


if __name__ == "__main__":
    main()

