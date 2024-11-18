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
import tkinter as tk
from tkinter.scrolledtext import ScrolledText




def log_message(message, text_widget):
    print(message)  # Output to terminal
    text_widget.insert(tk.END, message + "\n")
    text_widget.see(tk.END)

def print_format(data):
    return "\n".join(f"{key}: {value}" for key, value in data.items())

def update_list_memory(original_list, detected_data, text_widget):
    log_message("Original list:\n" + print_format(original_list), text_widget)

    for name, count in detected_data.items():
        if name not in original_list:
            log_message(f"{name} 추가.", text_widget)
            original_list[name] = count
        elif original_list[name] != count:
            log_message(f"{name} 수량 업데이트.", text_widget)
            original_list[name] = count
        else:
            print(f"{name} 수량 일치.")

    return original_list

def create_gui(simulation_function):
    # Create a GUI window
    root = tk.Tk()
    root.title("Real-Time List Update Viewer")

    # Add a ScrolledText widget to display updates
    text_widget = ScrolledText(root, wrap=tk.WORD, width=50, height=20)
    text_widget.pack(padx=10, pady=10)

    # Start the GUI loop
    root.mainloop()

@torch.no_grad()
def run(
        weights, source, data, imgsz, conf_thres, iou_thres, max_det, device,
        save_txt, save_conf, save_crop, nosave, classes, agnostic_nms,
        augment, visualize, project, name, exist_ok, line_thickness,
        hide_labels, hide_conf, half, dnn
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Set up source, 뽑을 곳.
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 스트리밍 URL이나 웹캠이면 True, 이미지/비디오파일 등이면 False
    webcam = source.startswith(('rtsp://', 'http://', 'https://')) or source.isnumeric()
    if is_file:
        source = check_file(source)

    # Set up directories, 저장할 곳.
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Set up dataloader, webcam = 1이면 받아옴.
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) if webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1 if not webcam else len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    detected_classes = {}
    detected_frame = None

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        t2 = time_sync()

        pred = model(im, augment=augment) # 객체 탐지

        t3 = time_sync()

        # 중복 박스 제거, 최종 탐지결과 반환
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                detected_frame = im0s  # 탐지된 프레임 저장 (마지막 프레임)

                # 탐지된 객체를 딕셔너리로 저장
                for *xyxy, conf, cls in det:
                    class_name = names[int(cls)]
                    detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
    
    # 탐지된 프레임이 있는 경우에만 저장 및 표시
    if detected_frame is not None and detected_classes:
        save_path = os.path.expanduser("~/detected_frame.jpg")
        cv2.imwrite(save_path, detected_frame)
        print(f"Saved detected frame to {save_path}")

        # 이미지 화면에 띄우기
        cv2.imshow("Detected Frame", detected_frame)  # OpenCV 창에 이미지 표시
        cv2.waitKey(0)  # 키 입력 대기 (0: 무한 대기)
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
    
    return detected_classes

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

def detection_callback(msg):
    print("Received detection command. Running YOLO on live stream...")
    
    # YOLO 실행 옵션 설정
    opt = parse_opt()
    opt.source = 0  # 카메라 스트리밍 URL
    opt.imgsz = [320, 320]  # 이미지 크기
    opt.conf_thres = 0.25  # Confidence threshold
    opt.iou_thres = 0.45  # IoU threshold for NMS
    opt.device = '0'  # GPU 사용
    opt.classes = None  # 모든 클래스 감지
    opt.augment = False  # 증강 비활성화

    detected_classes = run(**vars(opt)) or []
    
    if not detected_classes:
        print("No detected classes. Continuing...")
    else:
        print(f"Detected classes: {detected_classes}")
    
    # 결과 비교 및 업데이트
    update_list_memory(detected_classes)
    print("YOLO detection on live stream completed.")

    # 객체 인식 완료 메시지 발행
    detection_done = rospy.Publisher('/detection_done', String, queue_size=10)
    detection_done.publish("Detection completed")

def main():
    rospy.init_node('yolo_detection_node', anonymous=True)

    # GUI 실행을 별도의 스레드에서 실행
    gui_thread = threading.Thread(target=create_gui, args=(simulate_detection_update,))
    gui_thread.daemon = True
    gui_thread.start()

    # 객체인식 실행 명령 토픽 구독 설정
    rospy.Subscriber('/start_detection', String, detection_callback)
    rospy.loginfo("YOLO detection node is ready and waiting for commands...")

    rospy.spin() # 메시지가 들어오면 콜백 실행

if __name__ == "__main__":
    main()

