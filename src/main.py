import numpy as np
import cv2
import threading
import queue
import time

from datetime import datetime


# 상수 정의
# 키 코드 상수
KEY_RECORD = 32  # SPACE  - 녹화 토글
KEY_STOP   = 27  # ESC    - 프로그램 종료
KEY_OUTPUT = 111 # o      - 녹화 파일 출력

# 효과 토글 키 상수
KEY_TOGGLE_LENS = 108      # l - 렌즈 왜곡 토글
KEY_TOGGLE_VIGNETTE = 118  # v - 비네트 토글
KEY_TOGGLE_SEPIA = 115     # s - 세피아 필터 토글
KEY_TOGGLE_VHS = 99        # c - VHS 컬러 시프트 토글
KEY_TOGGLE_NOISE = 110     # n - 노이즈 토글
KEY_TOGGLE_SCRATCHES = 120 # x - 스크래치 토글

# 디스플레이 상수
DISPLAY_TITLE = "HIDEO"
VIDEO_SOURCE = 0   # RTSP 주소 또는 0-N (카메라)
OS_CODEC = 'mp4v'  # macOS/Linux mp4v, Windows XVID
OS_CODEC_POSTFIX = 'mp4'
TEXT_COLOR = (255, 255, 255)
RECORD_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX # 기본 폰트만 사용가능함
FONT_SCALE = 0.7
FONT_THICKNESS = 2
s_FONT_SCALE = 0.5
s_FONT_THICKNESS = 1

# 효과 상수
DEFAULT_VIGNETTE_STRENGTH = 1.0
DEFAULT_VIGNETTE_DISTANCE_SCALE = 1.4
DEFAULT_LENS_DISTORTION_STRENGTH = 0.07
DEFAULT_SEPIA_STRENGTH = 0.5
DEFAULT_NOISE_BUFFER_SIZE = 5  # 노이즈 버퍼 크기 상수


class VideoProcessor:
    def __init__(self, input_path=0):
        # 입력 경로 (RTSP 또는 파일 경로)
        self.cap = cv2.VideoCapture(input_path)
        
        # 비디오 속성
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cv2.CAP_PROP_FPS
        
        # 코덱, 출력 객체 초기화
        self.fourcc = cv2.VideoWriter_fourcc(*OS_CODEC)
        self.out = None
        
        # 후처리 속성
        self.vignette_strength = DEFAULT_VIGNETTE_STRENGTH
        self.vignette_distance_scale = DEFAULT_VIGNETTE_DISTANCE_SCALE
        self.lens_distortion_strength = DEFAULT_LENS_DISTORTION_STRENGTH
        self.sepia_strength = DEFAULT_SEPIA_STRENGTH

        # 가우시안 노이즈 버퍼 생성 (초기화 시 한 번만 계산하여 메모리에 저장)
        self.noise_buffer = self._create_gaussian_noise(self.frame_width, self.frame_height,
                                                       mean=0, sigma=25, buffer_size=DEFAULT_NOISE_BUFFER_SIZE)
        
        # 비네트 마스크 생성 (초기화 시 한 번만 계산하여 메모리에 저장)
        self.vignette_mask = self._create_vignette_mask(self.frame_width, self.frame_height,
                                                  strength=self.vignette_strength,
                                                  distance_scale=self.vignette_distance_scale)
        self.vignette_mask_3d = cv2.merge([self.vignette_mask, self.vignette_mask, self.vignette_mask])
        
        # 효과 활성화 플래그
        self.is_recording = False
        self.enable_lens_distortion = True
        self.enable_vignette = True
        self.enable_sepia = False
        self.enable_vhs = True
        self.enable_noise = True
        self.enable_scratches = False
        
        # 프레임 계산
        self.record_frame_count = 0  # 녹화 프레임 카운트
        self.total_frame_count = 0   # 전체 프레임 카운트 (노이즈 효과 등에 사용)
        
        # FPS 조절을 위한 타이머 변수
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.fps
    
    def _add_scratches(self, frame, num_scratches=3):
        """
        랜덤한 아날로그 테이프 스크래치 효과를 추가합니다.
        """
        scratched_frame = frame.copy()
        height, width = frame.shape[:2]
        for _ in range(num_scratches):
            x = np.random.randint(0, width)  # 랜덤 x 위치
            thickness = np.random.randint(1, 3)  # 선 두께
            cv2.line(scratched_frame, (x, 0), (x, height), (200, 200, 200), thickness)  # 밝은 선
        return scratched_frame
    
    def _create_gaussian_noise(self, width, height, mean=0, sigma=25, buffer_size=5):
        """
        여러 개의 노이즈 프레임을 미리 생성하여 버퍼에 저장합니다.
        성능 향상을 위해 매번 노이즈를 생성하지 않고 버퍼에서 순환하여 사용합니다.
        """
        # 여러 노이즈 프레임을 버퍼에 저장
        noise_buffer = []
        for _ in range(buffer_size):
            noise = np.random.normal(mean, sigma, (height, width, 3)).astype(np.float32)
            noise_buffer.append(noise)
        return noise_buffer
    
    def _add_gaussian_noise(self, frame, noise_buffer=None):
        """
        가우시안 노이즈를 화면에 추가합니다.
        버퍼에 저장된 노이즈 프레임을 순환하여 사용합니다.
        """
        if noise_buffer is None:
            noise_buffer = self.noise_buffer
            
        # 버퍼에서 현재 노이즈 프레임 선택 (순환)
        # 전체 프레임 카운트를 사용하여 녹화 여부와 상관없이 노이즈 변화
        buffer_idx = self.total_frame_count % len(noise_buffer)
        noise = noise_buffer[buffer_idx]
        
        # 프레임에 노이즈 추가
        noisy_frame = frame.astype(np.float32) + noise
        # 0~255 범위로 클리핑
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        return noisy_frame
    
    def _apply_vhs_color_shift(self, frame, shift_amount=2):
        """
        Red, Blue 채널을 분리하는 효과를 줍니다.
        :param frame: 입력 프레임 (numpy 배열)
        :param shift_amount: 채널 이동 정도 (기본값: 2)
        :return: 효과가 적용된 프레임
        """
        # 채널 분리
        b, g, r = cv2.split(frame)
        # 각 채널에 약간의 오프셋 추가
        b_shifted = np.roll(b, shift_amount, axis=1)  # 파란색 약간 오른쪽으로 이동
        r_shifted = np.roll(r, -shift_amount, axis=1)  # 빨간색 약간 왼쪽으로 이동
        # 다시 합침
        vhs_frame = cv2.merge([b_shifted, g, r_shifted])
        return vhs_frame
    
    def _apply_sepia(self, frame, strength=1.0):
        """
        오래된 느낌을 주는 색조 (세피아)
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
    
    def _create_vignette_mask(self, width, height, strength=1.0, distance_scale=1.0):
        """비네트 효과 마스크 생성 메소드"""
        if width == 0 or height == 0:  # 카메라 초기화 실패 시 대비
            return np.ones((1, 1), dtype=np.uint8) * 255
            
        kernel_x = np.linspace(-1, 1, width)
        kernel_y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(kernel_x, kernel_y)
        D = np.sqrt(X**2 + Y**2)  # Distance from center
        D_scaled = D / distance_scale  # 거리 스케일링으로 효과 범위 조정
        mask = np.clip(1 - D_scaled * strength, 0, 1)  # Inverse of scaled distance
        
        # np.max(mask)가 0인 경우(빈 마스크) 예외 처리
        max_val = np.max(mask)
        if max_val > 0:
            mask = (mask / max_val * 255).astype(np.uint8)  # Scale to 255
        else:
            mask = np.ones_like(mask, dtype=np.uint8) * 255
            
        return mask
    
    def _apply_vignette(self, frame, mask_3d):
        """비네트 효과 적용 메소드"""
        return (frame * (mask_3d / 255)).astype(np.uint8)
    
    def _apply_lens_distortion(self, frame, strength=0.2, zoom_factor=1.2):
        """
        렌즈 왜곡 메소드
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
        
        return cropped_frame
    
    def _add_overlay_text(self, frame):
        """화면에 오버레이 텍스트 추가 (비율 기반)"""
        # 현재 시간
        current_time = datetime.now()
        time_text = current_time.strftime("%H:%M:%S")
        
        # 비율 기반 위치 계산 (화면 크기 변경에 일관성 유지)
        w, h = self.frame_width, self.frame_height
        
        # 위치 간격 및 여백 (비율 기반) - 렌즈 왜곡 효과를 고려해 더 안쪽으로 배치
        margin_ratio = 0.15  # 화면 가장자리로부터의 여백 (15%)
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        
        # 녹화 중 표시
        blink = (self.record_frame_count // int(self.fps / 2)) % 2  # FPS 기반 0.5초 주기
        
        # 하단 왼쪽 (녹화 상태) - 렌즈 왜곡 고려하여 안쪽으로 배치
        bottom_left_x = margin_x
        bottom_left_y = int(h * 0.85)  # 하단에서 15% 위치
        
        # 녹화 중임을 나타내는 원
        if self.is_recording:
            if blink:
                # 원 위치 (하단 왼쪽)
                record_circle_x = bottom_left_x
                record_circle_y = bottom_left_y - int(h * 0.01)  # 작은 조정
                cv2.circle(frame, (record_circle_x, record_circle_y), 
                          int(h * 0.01), RECORD_COLOR, -1)  # 높이의 1% 크기
            record_status_text = "RECORD"
        else:
            record_status_text = "PAUSED"
        
        # 녹화 상태 텍스트 위치
        record_text_x = bottom_left_x + int(w * 0.02)  # 원 오른쪽으로 2% 이동
        record_text_y = bottom_left_y
        cv2.putText(frame, record_status_text, (record_text_x, record_text_y),
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 녹화 시간 및 프레임
        elapsed_time_seconds = self.record_frame_count / self.fps
        elapsed_time_minutes = int(elapsed_time_seconds // 60)
        elapsed_time_seconds = int(elapsed_time_seconds % 60)
        elapsed_time_text = f"TIME {elapsed_time_minutes:02}:{elapsed_time_seconds:02}"
        
        # 하단 오른쪽 (녹화 정보)
        bottom_right_x = int(w * (1 - margin_ratio))
        bottom_right_y = bottom_left_y
        
        # 텍스트 길이 측정 및 오른쪽 정렬을 위한 보정
        frame_text = f"FRAME {self.record_frame_count}/{self.total_frame_count}"
        
        # 녹화 프레임 텍스트 위치 (하단 오른쪽)
        frame_text_size = cv2.getTextSize(frame_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        frame_text_x = bottom_right_x - frame_text_size[0]
        frame_text_y = bottom_right_y
        cv2.putText(frame, frame_text, (frame_text_x, frame_text_y),
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 녹화 시간 텍스트 위치 (프레임 위)
        time_text_size = cv2.getTextSize(elapsed_time_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        time_text_x = bottom_right_x - time_text_size[0]
        time_text_y = frame_text_y - int(h * 0.03)  # 위로 3% 이동
        cv2.putText(frame, elapsed_time_text, (time_text_x, time_text_y),
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 상단 영역 - 렌즈 왜곡 고려하여 더 아래로 배치
        top_margin_y = margin_y + int(h * 0.05)  # 상단에서 5% 아래로
        
        # 현재 시간 표시 (상단 중앙)
        time_text_size = cv2.getTextSize(time_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        current_time_x = (w - time_text_size[0]) // 2  # 중앙 정렬
        current_time_y = top_margin_y
        cv2.putText(frame, time_text, (current_time_x, current_time_y),
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # HIDEO 로고 (상단 오른쪽)
        logo_text = "HIDEO"
        logo_text_size = cv2.getTextSize(logo_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        logo_x = bottom_right_x - logo_text_size[0]
        logo_y = top_margin_y
        cv2.putText(frame, logo_text, (logo_x, logo_y),
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 활성화된 효과 표시 (상단 왼쪽)
        effects_status = []
        if self.enable_lens_distortion:
            effects_status.append("LENS")
        if self.enable_vignette:
            effects_status.append("VIGN")
        if self.enable_sepia:
            effects_status.append("SEPIA")
        if self.enable_vhs:
            effects_status.append("VHS")
        if self.enable_noise:
            effects_status.append("NOISE")
        if self.enable_scratches:
            effects_status.append("SCRCH")
        
        effects_text = " ".join(effects_status)
        effects_x = bottom_left_x
        effects_y = top_margin_y
        cv2.putText(frame, effects_text, (effects_x, effects_y),
                   FONT, s_FONT_SCALE, TEXT_COLOR, s_FONT_THICKNESS)
        
        return frame
    
    def _apply_effects(self, frame):
        """모든 효과를 프레임에 적용"""
        # 후처리를 위해 프레임 복사
        processed_frame = frame.copy()
        
        # 색조 변경 (세피아)
        if self.enable_sepia:
            processed_frame = self._apply_sepia(processed_frame, self.sepia_strength)
        
        # VHS 컬러 시프트
        if self.enable_vhs:
            processed_frame = self._apply_vhs_color_shift(processed_frame)
        
        # 노이즈 효과
        if self.enable_noise:
            processed_frame = self._add_gaussian_noise(processed_frame, self.noise_buffer)
        
        # 스크래치 효과
        if self.enable_scratches:
            processed_frame = self._add_scratches(processed_frame, 3)
        
        # 비네트 효과 적용 (강한 효과)
        if self.enable_vignette:
            processed_frame = self._apply_vignette(processed_frame, self.vignette_mask_3d)
        
        # 텍스트 및 오버레이 추가
        processed_frame = self._add_overlay_text(processed_frame)
        
        # 비네트 적용 (약한 효과) - 항상 적용
        processed_frame = (processed_frame * (self.vignette_mask_3d / 255) * 0.25 + processed_frame * 0.75).astype(np.uint8)
        
        # 렌즈 효과 적용
        if self.enable_lens_distortion:
            processed_frame = self._apply_lens_distortion(processed_frame, self.lens_distortion_strength)
        
        return processed_frame
    
    def toggle_recording(self):
        """녹화 시작/중지 토글"""
        self.is_recording = not self.is_recording
        
        if self.is_recording and self.out is None:
            output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{OS_CODEC_POSTFIX}"
            self.out = cv2.VideoWriter(output_filename, self.fourcc, self.fps, 
                                      (self.frame_width, self.frame_height))
            print("녹화 시작")
            # 녹화 시작 시 프레임 카운트 초기화
            self.record_frame_count = 0
            self.last_frame_time = time.time()
        elif not self.is_recording and self.out is not None:
            print("녹화 일시정지")
    
    def release_output(self):
        """녹화된 영상 출력 및 초기화"""
        if self.out is not None:
            self.out.release()
            self.out = None
            print("녹화 종료 및 파일 저장")
    
    def process_frame(self):
        """프레임 처리 및 반환"""
        ret, frame = self.cap.read()
        if not ret:
            print("소스를 읽을 수 없습니다.")
            return None
        
        # 전체 프레임 카운트 증가 (효과 적용 전에 업데이트)
        self.total_frame_count += 1
        
        # 모든 효과 적용
        processed_frame = self._apply_effects(frame)
        
        # 녹화 중이면 FPS에 맞춰 프레임 저장
        if self.is_recording and self.out is not None:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            
            # FPS에 맞게 시간이 경과했으면 프레임 저장
            if elapsed >= self.frame_interval:
                self.out.write(processed_frame)
                self.record_frame_count += 1
                self.last_frame_time = current_time
        
        return processed_frame
    
    def toggle_effect(self, effect_name):
        """효과 토글 메소드"""
        if effect_name == "lens":
            self.enable_lens_distortion = not self.enable_lens_distortion
            return f"렌즈 왜곡: {'켜짐' if self.enable_lens_distortion else '꺼짐'}"
        elif effect_name == "vignette":
            self.enable_vignette = not self.enable_vignette
            return f"비네트: {'켜짐' if self.enable_vignette else '꺼짐'}"
        elif effect_name == "sepia":
            self.enable_sepia = not self.enable_sepia
            return f"세피아: {'켜짐' if self.enable_sepia else '꺼짐'}"
        elif effect_name == "vhs":
            self.enable_vhs = not self.enable_vhs
            return f"VHS 컬러 시프트: {'켜짐' if self.enable_vhs else '꺼짐'}"
        elif effect_name == "noise":
            self.enable_noise = not self.enable_noise
            return f"노이즈: {'켜짐' if self.enable_noise else '꺼짐'}"
        elif effect_name == "scratches":
            self.enable_scratches = not self.enable_scratches
            return f"스크래치: {'켜짐' if self.enable_scratches else '꺼짐'}"
        return "알 수 없는 효과"
    
    def handle_key_press(self, key):
        """키 입력 처리"""
        if key == KEY_RECORD:
            self.toggle_recording()
            return True
        elif key == KEY_OUTPUT and self.out is not None:
            self.release_output()
            return True
        elif key == KEY_STOP:
            return False
        # 효과 토글 키 처리
        elif key == KEY_TOGGLE_LENS:
            print(self.toggle_effect("lens"))
            return True
        elif key == KEY_TOGGLE_VIGNETTE:
            print(self.toggle_effect("vignette"))
            return True
        elif key == KEY_TOGGLE_SEPIA:
            print(self.toggle_effect("sepia"))
            return True
        elif key == KEY_TOGGLE_VHS:
            print(self.toggle_effect("vhs"))
            return True
        elif key == KEY_TOGGLE_NOISE:
            print(self.toggle_effect("noise"))
            return True
        elif key == KEY_TOGGLE_SCRATCHES:
            print(self.toggle_effect("scratches"))
            return True
        return True
    
    def release_resources(self):
        """메모리 반환"""
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()


def main():
    # 비디오 프로세서 초기화
    processor = VideoProcessor(input_path=VIDEO_SOURCE)
    
    print("\n===== HIDEO 플레이어 =====")
    print("효과 토글 키:")
    print("l - 렌즈 왜곡 켜기/끄기")
    print("v - 비네트 켜기/끄기")
    print("s - 세피아 켜기/끄기")
    print("c - VHS 컬러 시프트 켜기/끄기")
    print("n - 노이즈 켜기/끄기")
    print("x - 스크래치 켜기/끄기")
    print("o - 녹화 종료 및 파일 저장")
    print("SPACE - 녹화 시작/일시정지")
    print("ESC - 프로그램 종료")
    print("=======================\n")
    
    # 메인 루프
    running = True
    while running:
        # 프레임 처리
        processed_frame = processor.process_frame()
        if processed_frame is None:
            break
        
        # 화면 출력
        cv2.imshow(DISPLAY_TITLE, processed_frame)
        
        # 키 입력 - 1ms 대기 (낮은 지연으로 인터페이스 응답성 유지)
        key = cv2.waitKey(1) & 0xFF
        running = processor.handle_key_press(key)
        
        # CPU 부하 감소를 위한 작은 지연
        time.sleep(0.001)
    
    # 메모리 반환
    processor.release_resources()


if __name__ == "__main__":
    main()

# TODO
# 프레임을 타임스탬프와 함께 저장
# 타임스탬프를 기반으로 영상을 녹화, 저장