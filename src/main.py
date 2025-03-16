import numpy as np
import cv2
from datetime import datetime

# 상수 정의
# 키 코드 상수
KEY_RECORD = 32  # SPACE
KEY_STOP = 113   # q
KEY_OUTPUT = 27  # ESC

# 디스플레이 상수
DISPLAY_TITLE = "HIDEO"
TEXT_COLOR = (255, 255, 255)
RECORD_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# 효과 상수
DEFAULT_VIGNETTE_STRENGTH = 1.0
DEFAULT_VIGNETTE_DISTANCE_SCALE = 1.4
DEFAULT_LENS_DISTORTION_STRENGTH = 0.07


class VideoProcessor:
    def __init__(self, input_path=0, fps=12.0):
        # 입력 경로 (RTSP 또는 파일 경로)
        self.cap = cv2.VideoCapture(input_path)
        
        # 비디오 속성
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        # 코덱, 출력 객체 초기화
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # macOS/Linux mp4v, Windows XVID
        self.out = None
        
        # 후처리 속성
        self.vignette_strength = DEFAULT_VIGNETTE_STRENGTH
        self.vignette_distance_scale = DEFAULT_VIGNETTE_DISTANCE_SCALE
        self.lens_distortion_strength = DEFAULT_LENS_DISTORTION_STRENGTH
        
        # 비네트 마스크 생성
        self.vignette_mask = self._create_vignette_mask(self.frame_width, self.frame_height,
                                                  strength=self.vignette_strength,
                                                  distance_scale=self.vignette_distance_scale)
        self.vignette_mask_3d = cv2.merge([self.vignette_mask, self.vignette_mask, self.vignette_mask])
        
        # 플래그
        self.is_recording = False
        self.enable_lens_distortion = True
        self.enable_vignette = True
        
        # 프레임 계산
        self.frame_count = 0
    
    def _add_scratches(self, frame, num_scratches=3):
        """아날로그 수직 선 추가 메소드"""
        scratched_frame = frame.copy()
        height, width = frame.shape[:2]
        for _ in range(num_scratches):
            x = np.random.randint(0, width)  # 랜덤 x 위치
            thickness = np.random.randint(1, 3)  # 선 두께
            cv2.line(scratched_frame, (x, 0), (x, height), (200, 200, 200), thickness)  # 밝은 선
        return scratched_frame
    
    def _add_gaussian_noise(self, frame, mean=0, sigma=25):
        """랜덤한 화이트 노이즈 추가 메소드"""
        # 가우시안 노이즈 생성
        noise = np.random.normal(mean, sigma, frame.shape).astype(np.float32)
        # 프레임에 노이즈 추가
        noisy_frame = frame.astype(np.float32) + noise
        # 0~255 범위로 클리핑
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        return noisy_frame
    
    def _apply_vhs_color_shift(self, frame):
        """VHS 컬러 분리 메소드"""
        # 채널 분리
        b, g, r = cv2.split(frame)
        # 각 채널에 약간의 오프셋 추가
        b_shifted = np.roll(b, 2, axis=1)  # 파란색 약간 오른쪽으로 이동
        r_shifted = np.roll(r, -2, axis=1)  # 빨간색 약간 왼쪽으로 이동
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
        kernel_x = np.linspace(-1, 1, width)
        kernel_y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(kernel_x, kernel_y)
        D = np.sqrt(X**2 + Y**2)  # Distance from center
        D_scaled = D / distance_scale  # 거리 스케일링으로 효과 범위 조정
        mask = np.clip(1 - D_scaled * strength, 0, 1)  # Inverse of scaled distance
        mask = (mask / np.max(mask) * 255).astype(np.uint8)  # Scale to 255
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
        """화면에 오버레이 텍스트 추가"""
        # 현재 시간
        current_time = datetime.now()
        time_text = current_time.strftime("%H:%M:%S")
        
        # 녹화 중 표시
        blink = (self.frame_count // int(self.fps / 2)) % 2  # FPS 기반 0.5초 주기
        
        # 녹화 중임을 나타내는 원
        if self.is_recording:
            if blink:
                cv2.circle(frame, (75, self.frame_height - 60), 8, RECORD_COLOR, -1)  # 빨간 원
            record_status_text = "RECORD"
        else:
            record_status_text = "PAUSED"
        
        cv2.putText(frame, record_status_text, (90, self.frame_height - 53),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 녹화 시간 및 프레임
        elapsed_time_seconds = self.frame_count / self.fps
        elapsed_time_minutes = int(elapsed_time_seconds // 60)
        elapsed_time_seconds = int(elapsed_time_seconds % 60)
        elapsed_time_text = f"TIME {elapsed_time_minutes:02}:{elapsed_time_seconds:02}"
        
        cv2.putText(frame, f"FRAME {self.frame_count}", (self.frame_width - 200, self.frame_height - 53),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(frame, elapsed_time_text, (self.frame_width - 200, self.frame_height - 73),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 현재 시간 표시
        cv2.putText(frame, time_text, (270, 75),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # HIDEO 로고
        cv2.putText(frame, "HIDEO", (self.frame_width - 150, 75),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        return frame
    
    def _apply_effects(self, frame):
        """모든 효과를 프레임에 적용"""
        # 후처리를 위해 프레임 복사
        processed_frame = frame.copy()
        
        # 색조 변경 (필요시 주석 해제)
        # processed_frame = self._apply_sepia(processed_frame, 0.5)
        
        # 각종 아날로그 효과
        processed_frame = self._apply_vhs_color_shift(processed_frame)
        processed_frame = self._add_gaussian_noise(processed_frame)
        # processed_frame = self._add_scratches(processed_frame, 3)
        
        # 비네트 효과 적용
        if self.enable_vignette:
            processed_frame = self._apply_vignette(processed_frame, self.vignette_mask_3d)
        
        # 텍스트 및 오버레이 추가
        processed_frame = self._add_overlay_text(processed_frame)
        
        # 비네트 적용 (약한 효과)
        processed_frame = (processed_frame * (self.vignette_mask_3d / 255) * 0.25 + processed_frame * 0.75).astype(np.uint8)
        
        # 렌즈 효과 적용
        if self.enable_lens_distortion:
            processed_frame = self._apply_lens_distortion(processed_frame, self.lens_distortion_strength)
        
        return processed_frame
    
    def toggle_recording(self):
        """녹화 시작/중지 토글"""
        self.is_recording = not self.is_recording
        
        if self.is_recording and self.out is None:
            output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            self.out = cv2.VideoWriter(output_filename, self.fourcc, self.fps, 
                                      (self.frame_width, self.frame_height))
            print("녹화 시작")
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
        
        # 모든 효과 적용
        processed_frame = self._apply_effects(frame)
        
        # 녹화 중이면 프레임 저장
        if self.is_recording and self.out is not None:
            self.out.write(processed_frame)
            self.frame_count += 1
        
        return processed_frame
    
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
        return True
    
    def release_resources(self):
        """메모리 반환"""
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()


def main():
    # 비디오 프로세서 초기화
    processor = VideoProcessor(input_path=0, fps=12.0)
    
    # 메인 루프
    running = True
    while running:
        # 프레임 처리
        processed_frame = processor.process_frame()
        if processed_frame is None:
            break
        
        # 화면 출력
        cv2.imshow(DISPLAY_TITLE, processed_frame)
        
        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        running = processor.handle_key_press(key)
    
    # 메모리 반환
    processor.release_resources()


if __name__ == "__main__":
    main()