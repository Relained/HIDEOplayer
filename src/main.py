import numpy as np
import cv2

input_path = 0
cap = cv2.VideoCapture(input_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = "test.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

KEY_RECORD = 32 # SPACE
KEY_STOP = 113 # q
KEY_OUTPUT = 27 # ESC

# 각종 플래그
recordFlag = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("There is no source available.")
        break
    
    key = cv2.waitKey(1) & 0xFF

    # 후처리


    # 프레임 녹화
    if key == KEY_RECORD:
        recordFlag = not recordFlag # 녹화 시작/중지 토글
        if recordFlag and out is None:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print("record start")
        elif not recordFlag and out is not None:
            print("record pause")
    
    # 녹화된 영상 출력
    elif key == KEY_OUTPUT and out is not None:
        out.release()
        out = None
        print("record released")

    # 화면 출력
    cv2.imshow('HIDEO', frame)

    # 프로그램 종료
    if cv2.waitKey(1) & 0xFF == KEY_STOP:
        break

# 자원 해제
cap.release()
if out: # 녹화중이던거 있으면 출력
    out.release()
cv2.destroyAllWindows()


# 로직:
# space를 눌렀을 때 녹화 시작 (플래그 사용)
# space를 눌렀을 때 녹화 일시정지
# esc를 눌렀을 때 녹화 종료 + 출력
# q를 눌렀을 때 프로그램 종료
# 메뉴 창에서 각종 효과 출력 (플래그 사용)