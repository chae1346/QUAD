11/16 update...

드론 (Jetson Nano)
 ├── ROS 노드: SLAM (RealSense)
 ├── ROS 노드: MAVROS (OFFBOARD 제어)
 ├── 카메라 스트리밍 서버 (GStreamer, OpenCV 등)
     └─> 노트북 (YOLO 실행)
          └─> 결과를 ROS 토픽으로 Jetson Nano에 전달
          
src/my_ros_pkg/offboard.py

1. 오프보드 코드가 목표 지점 도달 여부를 판단:
    SLAM으로부터 토픽을 구독하지 못했을 경우, MAVROS 데이터 및 키보드 인터럽트 사용.
    목표 지점으로 이동 및 도달 여부 확인.

2. 목표 지점 도달 시:
    호버링 유지.
    YOLO 실행 토픽 발행.

3. 노트북에서 객체 탐지 실행:
    객체 탐지 완료 알림 토픽 발행.

4. Jetson Nano가 객체 탐지 결과를 구독:
    후속 행동(예: 특정 객체로 이동, 다음 목표로 전환)을 수행.
    
    
src/yolo.../detect7.py

