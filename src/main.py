import numpy as np
import cv2
from datetime import datetime

# 입력 경로 (RTSP 또는 파일 경로)
input_path = 0
cap = cv2.VideoCapture(input_path)

# 비디오 속성
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 코덱, 출력 객체 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # macOS/Linux mp4v, Windows XVID
out = None

# 키 코드 상수 정의
KEY_RECORD = 32 # SPACE
KEY_STOP = 113 # q
KEY_OUTPUT = 27 # ESC

# 플래그
recordFlag = False


while True:
    # 초기 프레임 입력
    ret, frame = cap.read()
    if not ret:
        print("There is no source available.")
        break
    

    # 후처리
    proc_frame = frame.copy()



    # 키 입력
    key = cv2.waitKey(1) & 0xFF

    # 프레임 녹화 초기화
    if key == KEY_RECORD:
        recordFlag = not recordFlag # 녹화 시작/중지 토글
        if recordFlag and out is None:
            out = cv2.VideoWriter(f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                  fourcc, fps, (frame_width, frame_height))
            print("record start")
        elif not recordFlag and out is not None:
            print("record pause") # 디버깅용 출력
    
    # 녹화된 영상 출력 및 초기화
    elif key == KEY_OUTPUT and out is not None:
        out.release()
        out = None
        print("record released")

    # 프로그램 종료
    elif key == KEY_STOP:
        break
    

    # 화면 출력
    cv2.imshow('HIDEO', proc_frame)

    # 녹화
    if recordFlag and out is not None:
        out.write(proc_frame)
    

# 메모리 반환
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()