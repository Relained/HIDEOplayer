import numpy as np
import cv2
from datetime import datetime

# 입력 경로 (RTSP 또는 파일 경로)
input_path = 0
cap = cv2.VideoCapture(input_path)

# 비디오 속성
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
fps = 12.0

# 코덱, 출력 객체 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # macOS/Linux mp4v, Windows XVID
out = None

# 키 코드 상수 정의
KEY_RECORD = 32 # SPACE
KEY_STOP = 113 # q
KEY_OUTPUT = 27 # ESC

# 후처리 속성
vignette_strength = 1.0
vignette_distance_scale = 1.4
lens_distortion_strength = 0.07


# 아날로그 수직 선 추가 메소드
def add_scratches(frame, num_scratches=3):
    scratched_frame = frame.copy()
    height, width = frame.shape[:2]
    for _ in range(num_scratches):
        x = np.random.randint(0, width)  # 랜덤 x 위치
        thickness = np.random.randint(1, 3)  # 선 두께
        cv2.line(scratched_frame, (x, 0), (x, height), (200, 200, 200), thickness)  # 밝은 선
    return scratched_frame

# 랜덤한 화이트 노이즈 추가 메소드
def add_gaussian_noise(frame, mean=0, sigma=25):
    # 가우시안 노이즈 생성
    noise = np.random.normal(mean, sigma, frame.shape).astype(np.float32)
    # 프레임에 노이즈 추가
    noisy_frame = frame.astype(np.float32) + noise
    # 0~255 범위로 클리핑
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    return noisy_frame

# VHS 컬러 분리 메소드
def apply_vhs_color_shift(frame):
    # 채널 분리
    b, g, r = cv2.split(frame)
    # 각 채널에 약간의 오프셋 추가
    b_shifted = np.roll(b, 2, axis=1)  # 파란색 약간 오른쪽으로 이동
    r_shifted = np.roll(r, -2, axis=1) # 빨간색 약간 왼쪽으로 이동
    # 다시 합침
    vhs_frame = cv2.merge([b_shifted, g, r_shifted])
    return vhs_frame

# 오래된 느낌을 주는 색조 (세피아)
def apply_sepia(frame, strength=1.0):
    """
    frame: 원본 프레임
    strength: 세피아 강도 (0.0: 원본, 1.0: 완전 세피아, 기본값 1.0)
    """
    # 세피아 변환 행렬
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    
    # 세피아 효과 적용
    sepia_frame = cv2.transform(frame, sepia_matrix)
    sepia_frame = np.clip(sepia_frame, 0, 255).astype(np.uint8)
    
    # 강도 조절 (알파 블렌딩)
    strength = np.clip(strength, 0, 1)  # 0~1 범위 제한
    blended_frame = cv2.addWeighted(frame, 1 - strength, sepia_frame, strength, 0.0)
    
    return blended_frame

# 비네트 효과 메소드 (거리 조절 추가)
def create_vignette_mask(width, height, strength=1.0, distance_scale=1.0):
    kernel_x = np.linspace(-1, 1, width)
    kernel_y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(kernel_x, kernel_y)
    D = np.sqrt(X**2 + Y**2)  # Distance from center
    D_scaled = D / distance_scale  # 거리 스케일링으로 효과 범위 조정
    mask = np.clip(1 - D_scaled * strength, 0, 1)  # Inverse of scaled distance
    mask = (mask / np.max(mask) * 255).astype(np.uint8)  # Scale to 255
    return mask

# 비네트 마스크 생성 (거리를 멀리 두기 위해 distance_scale 증가)
vignette_mask = create_vignette_mask(frame_width, frame_height,
                                    strength=vignette_strength, 
                                    distance_scale=vignette_distance_scale)
vignette_mask_3d = cv2.merge([vignette_mask, vignette_mask, vignette_mask])

# 렌즈 왜곡 메소드 (확대 및 리사이즈 추가)
def apply_lens_distortion(frame, strength=0.2, zoom_factor=1.2):
    """
    frame: 원본 프레임
    strength: 왜곡 강도 (기본값 0.2)
    zoom_factor: 확대 비율 (기본값 1.2, 1.0은 확대 없음)
    """
    height, width = frame.shape[:2]
    
    # 1. 왜곡 좌표 생성
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)  # Radius from center
    factor = 1 + strength * (r ** 2)  # Distortion factor
    X_distorted = X * factor
    Y_distorted = Y * factor

    # 2. 좌표를 픽셀 인덱스로 변환 (확대 고려)
    X_distorted = np.clip((X_distorted + 1) * 0.5 * (width - 1), 0, width - 1).astype(np.float32)
    Y_distorted = np.clip((Y_distorted + 1) * 0.5 * (height - 1), 0, height - 1).astype(np.float32)

    # 3. 왜곡 맵 생성 및 렌즈 왜곡 적용
    map_x = X_distorted
    map_y = Y_distorted
    distorted_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # 4. 확대 (가장자리 제거)
    zoomed_height, zoomed_width = int(height * zoom_factor), int(width * zoom_factor)
    zoomed_frame = cv2.resize(distorted_frame, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
    
    # 5. 중심에서 원래 크기로 크롭
    start_y = (zoomed_height - height) // 2
    start_x = (zoomed_width - width) // 2
    cropped_frame = zoomed_frame[start_y:start_y + height, start_x:start_x + width]

    # 6. 결과 반환 (필요 시 리사이즈 생략 가능)
    return cropped_frame

# 플래그
recordFlag = False
lenzDistortionFlag = True
viginetteFlag = True

# 프레임 계산
frameCount = 0
start_time = None
while True:
    # 초기 프레임 입력
    ret, frame = cap.read()
    if not ret:
        print("There is no source available.")
        break


    # 후처리
    proc_frame = frame.copy()

    # 색조 변경
    # proc_frame = apply_sepia(proc_frame, 0.5)

    # 각종 아날로그 효과
    proc_frame = apply_vhs_color_shift(proc_frame)
    proc_frame = add_gaussian_noise(proc_frame)
    # proc_frame = add_scratches(proc_frame, 3)
    
    # 비네트 효과 적용
    proc_frame = (proc_frame * (vignette_mask_3d / 255)).astype(np.uint8)

    # 녹화 중 표시
    blink = (frameCount // int(fps / 2)) % 2  # FPS 기반 0.5초 주기
    # 녹화 중임을 나타내는 원
    if recordFlag:
        if blink:
            cv2.circle(proc_frame, (75, frame_height - 60), 8, (0, 0, 255), -1)  # 빨간 원
        record_status_text = "RECORD"
    else:
        record_status_text = "PAUSED"
    
    cv2.putText(proc_frame, record_status_text, (90, frame_height - 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 흰색 텍스트

    # 녹화 시간 및 프레임
    elapsed_time_seconds = frameCount / fps  # Calculate elapsed time
    elapsed_time_minutes = int(elapsed_time_seconds // 60)  # Get minutes
    elapsed_time_seconds = int(elapsed_time_seconds % 60)  # Get seconds
    elapsed_time_text = f"TIME {elapsed_time_minutes:02}:{elapsed_time_seconds:02}"  # Format time

    cv2.putText(proc_frame, f"FRAME {frameCount}", (frame_width - 200, frame_height - 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Frame count text
    cv2.putText(proc_frame, elapsed_time_text, (frame_width - 200, frame_height - 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Time text

    proc_frame = (proc_frame * (vignette_mask_3d / 255) * 0.25 + proc_frame * 0.75).astype(np.uint8)

    # 현재 시간
    current_time = datetime.now()
    time_text = current_time.strftime("%H:%M:%S")

    cv2.putText(proc_frame, time_text, (270, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Time text
    
    # HIDEO
    cv2.putText(proc_frame, "HIDEO", (frame_width - 150, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        
    # 렌즈 효과 적용
    proc_frame = apply_lens_distortion(proc_frame, lens_distortion_strength)

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

    if recordFlag and out is not None:
        out.write(proc_frame)
        frameCount += 1

# 메모리 반환
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()