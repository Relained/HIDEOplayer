import numpy as np
import cv2
from datetime import datetime
import time

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
VIDEO_FPS = 60.0
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


class VideoProcessor:
    def __init__(self, input_path=0, fps=12.0):
        # 입력 경로 (RTSP 또는 파일 경로)
        self.cap = cv2.VideoCapture(input_path)
        
        # 비디오 속성
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        # 코덱, 출력 객체 초기화
        self.fourcc = cv2.VideoWriter_fourcc(*OS_CODEC)
        self.out = None
        
        # 후처리 속성
        self.vignette_strength = DEFAULT_VIGNETTE_STRENGTH
        self.vignette_distance_scale = DEFAULT_VIGNETTE_DISTANCE_SCALE
        self.lens_distortion_strength = DEFAULT_LENS_DISTORTION_STRENGTH
        self.sepia_strength = DEFAULT_SEPIA_STRENGTH

        # 가우시안 노이즈 패치 생성 (초기화 시 한 번만 계산하여 메모리에 저장)
        self.gaussian_noise = self._create_gaussian_noise(self.frame_width, self.frame_height,
                                                          mean=0, sigma=25, patch_size=128)
        
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
        self.frame_count = 0
        
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
    
    def _create_gaussian_noise(self, width, height, mean=0, sigma=25, patch_size=128):
        """
        작은 사이즈의 노이즈 패치를 생성합니다.
        이후 타일링하여 사용하기 위한 기본 패치입니다.
        """
        # 작은 노이즈 패치 생성 (성능 향상을 위해)
        noise_patch = np.random.normal(mean, sigma, (patch_size, patch_size, 3)).astype(np.float32)
        return noise_patch
    
    def _add_gaussian_noise(self, frame, noise_patch=None):
        """
        가우시안 노이즈를 화면에 추가합니다.
        타일링 방식으로 노이즈를 효율적으로 적용합니다.
        """
        if noise_patch is None:
            noise_patch = self.gaussian_noise
            
        height, width = frame.shape[:2]
        patch_size = noise_patch.shape[0]
        
        # 프레임 크기에 맞게 패치를 타일링
        h_tiles = int(np.ceil(height / patch_size))
        w_tiles = int(np.ceil(width / patch_size))
        
        # 전체 노이즈 맵 초기화
        noise_map = np.zeros(frame.shape, dtype=np.float32)
        
        # 패치 타일링
        for i in range(h_tiles):
            for j in range(w_tiles):
                h_start = i * patch_size
                w_start = j * patch_size
                h_end = min(h_start + patch_size, height)
                w_end = min(w_start + patch_size, width)
                
                # 패치의 일부 무작위 회전 또는 뒤집기 (다양성 추가)
                curr_patch = noise_patch.copy()
                if np.random.rand() > 0.5:
                    curr_patch = np.fliplr(curr_patch)
                if np.random.rand() > 0.5:
                    curr_patch = np.flipud(curr_patch)
                
                # 노이즈 맵에 패치 적용
                patch_h = h_end - h_start
                patch_w = w_end - w_start
                noise_map[h_start:h_end, w_start:w_end] = curr_patch[:patch_h, :patch_w]
        
        # 프레임에 노이즈 추가
        noisy_frame = frame.astype(np.float32) + noise_map
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
        
        # 활성화된 효과 표시
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
        cv2.putText(frame, effects_text, (75, 75),
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
            processed_frame = self._add_gaussian_noise(processed_frame, self.gaussian_noise)
        
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
            self.frame_count = 0
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
        
        # 모든 효과 적용
        processed_frame = self._apply_effects(frame)
        
        # 녹화 중이면 FPS에 맞춰 프레임 저장
        if self.is_recording and self.out is not None:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            
            # FPS에 맞게 시간이 경과했으면 프레임 저장
            if elapsed >= self.frame_interval:
                self.out.write(processed_frame)
                self.frame_count += 1
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
    processor = VideoProcessor(input_path=VIDEO_SOURCE, fps=VIDEO_FPS)
    
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