import cv2
import numpy as np
import time
import random
from enum import Enum


# ─── CONFIG ────────────────────────────────────────────────────────
CONFIG = {
    'frame_size': (640, 480),
    'blur_kernel': (5, 5),
    'green_move_min': 0.04,
    'red_move_threshold': 0.055,
    'warning_after_idle_sec': 1.8,
    'death_after_idle_sec': 3.6,
    'red_grace_period_sec': 0.65,
    'cycles_per_level': 3,
    'green_duration_range': (2.6, 4.2),
    'red_duration_range': (1.7, 2.9),
}


class GamePhase(Enum):
    GREEN = "GREEN"
    RED = "RED"
    WARN = "WARNING"
    DEAD = "DEAD"


class RedLightGreenLight:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_size'][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_size'][1])
        
        self.prev_gray = None
        self.phase = GamePhase.GREEN
        self.level = 1
        self.cycle = 0
        
        self.phase_start_time = time.time()
        self.green_duration = self._random_green_time()
        self.red_duration = self._random_red_time()
        
        self.idle_start = None
        self.see_person = False
        self.detector = None
        
        try:
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.pt')
        except:
            print("YOLO detector not available, skipping person detection")
    
    def _random_green_time(self):
        min_ms, max_ms = CONFIG['green_duration_range']
        return random.uniform(min_ms, max_ms)
    
    def _random_red_time(self):
        min_ms, max_ms = CONFIG['red_duration_range']
        return random.uniform(min_ms, max_ms)
    
    def _motion_level_threshold(self):
        # Motion threshold increases with level
        base = CONFIG['red_move_threshold']
        return base * (0.8 + 0.1 * self.level)
    
    def update_timers(self, now):
        elapsed = now - self.phase_start_time
        if self.phase == GamePhase.GREEN:
            max_duration = self.green_duration
        elif self.phase == GamePhase.RED:
            max_duration = self.red_duration
        else:
            max_duration = float('inf')
        return elapsed, max_duration
    
    def switch_phase(self, new_phase):
        self.phase = new_phase
        self.phase_start_time = time.time()
        self.idle_start = None
    
    def advance_cycle(self):
        self.cycle += 1
        if self.cycle >= CONFIG['cycles_per_level']:
            self.level += 1
            self.cycle = 0
            print(f"🎉 Advanced to LEVEL {self.level}!")
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, CONFIG['frame_size'])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, CONFIG['blur_kernel'], 0)

            # Motion calculation
            motion = 0.0
            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
                motion = np.mean(diff) / 255.0
            self.prev_gray = gray

            now = time.time()
            elapsed, max_duration = self.update_timers(now)

            # ─── Person detection (optional) ───
            self.see_person = False
            if self.detector is not None:
                results = self.detector(frame, verbose=False)
                for box in results[0].boxes:
                    if int(box.cls) == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
                        self.see_person = True
                        break

            # ─── GAME LOGIC ────────────────────────────────────────
            if self.phase == GamePhase.GREEN:
                if motion < CONFIG['green_move_min']:
                    if self.idle_start is None:
                        self.idle_start = now
                    idle_time = now - self.idle_start
                    if idle_time >= CONFIG['warning_after_idle_sec'] and self.phase != GamePhase.WARN:
                        self.switch_phase(GamePhase.WARN)
                    if idle_time >= CONFIG['death_after_idle_sec']:
                        self.switch_phase(GamePhase.DEAD)
                else:
                    self.idle_start = None

                if elapsed >= self.green_duration:
                    self.switch_phase(GamePhase.RED)
                    self.red_duration = self._random_red_time()
                    self.idle_start = None

            elif self.phase == GamePhase.WARN:
                if self.idle_start is None:
                    self.idle_start = now
                idle_time = now - self.idle_start

                if motion >= CONFIG['green_move_min']:
                    self.switch_phase(GamePhase.GREEN)
                elif idle_time >= CONFIG['death_after_idle_sec']:
                    self.switch_phase(GamePhase.DEAD)

            elif self.phase == GamePhase.RED:
                grace_passed = elapsed >= CONFIG['red_grace_period_sec']
                if grace_passed and motion > self._motion_level_threshold():
                    self.switch_phase(GamePhase.DEAD)

                if elapsed >= self.red_duration:
                    self.advance_cycle()
                    self.switch_phase(GamePhase.GREEN)
                    self.green_duration = self._random_green_time()

            # ─── DEAD phase: now just shows screen — no break! ───────
            if self.phase == GamePhase.DEAD:
                # Big red text
                cv2.putText(frame, "YOU DIED", (90, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 40, 255), 4)
                # Instructions
                cv2.putText(frame, "r = restart from level 1", (40, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 1)
                cv2.putText(frame, "q = quit the program", (40, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 1)
                # Still show motion & person box even when dead
                cv2.putText(frame, f"Motion: {motion:.3f}", (15, 440),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

            # ─── UI (shown in all phases) ─────────────────────────────
            color = {
                GamePhase.GREEN: (0, 255, 80),
                GamePhase.RED: (0, 0, 255),
                GamePhase.WARN: (40, 240, 240),
                GamePhase.DEAD: (60, 60, 255)
            }[self.phase]

            status = f"PHASE: {self.phase.value}"
            if self.phase == GamePhase.DEAD:
                status += "  (press r or q)"

            cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            time_left = max(0, max_duration - elapsed) if self.phase != GamePhase.DEAD else 0
            cv2.putText(frame, f"TIME: {time_left:4.1f}s", (380, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 180), 1)

            cv2.putText(frame, f"LEVEL {self.level}  CYCLE {self.cycle}/{CONFIG['cycles_per_level']}",
                        (380, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

            if self.phase == GamePhase.WARN:
                cv2.putText(frame, "KEEP MOVING!!", (90, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (80, 255, 255), 2)

            cv2.imshow("Red Light Green Light", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.phase == GamePhase.DEAD:
                # Full reset
                self.__init__()
                print("Game restarted — good luck!")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = RedLightGreenLight()
    print("Starting Red Light Green Light game...")
    print("Press 'r' to restart")
    print("Press 'q' to quit")
    game.run()