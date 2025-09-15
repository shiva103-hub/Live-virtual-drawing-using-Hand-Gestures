# hand_draw.py  -- fixed erase + camera stability issues

import os
import cv2
import time
import math
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Mediapipe is required. Install with: pip install mediapipe")

# -------------------- Utility helpers --------------------
class FPS:
    def __init__(self, avg_over=30):
        self.prev = time.time()
        self.avg_over = avg_over
        self.buf = []

    def tick(self):
        now = time.time()
        dt = now - self.prev
        self.prev = now
        if dt > 0:
            self.buf.append(1.0 / dt)
            if len(self.buf) > self.avg_over:
                self.buf.pop(0)
        return self.get()

    def get(self):
        return sum(self.buf) / len(self.buf) if self.buf else 0.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def timestamp_name(prefix="drawing", ext=".png"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"


# -------------------- Hand + gesture logic --------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
PIP_IDS = [3, 6, 10, 14, 18]


def fingers_up(hand_landmarks, handedness="Right"):
    """Return list of 5 booleans [thumb, index, middle, ring, pinky] indicating if each finger is up.
    Uses relative landmark positions. For thumb, use x-direction (differs by hand). Others use y.
    """
    lm = hand_landmarks
    up = [False]*5

    # Thumb: compare tip and IP x relative to MCP; flip by handedness
    thumb_tip = lm.landmark[4].x
    thumb_ip  = lm.landmark[3].x
    thumb_mcp = lm.landmark[2].x
    if handedness == "Right":
        up[0] = (thumb_tip < thumb_ip) and (thumb_ip < thumb_mcp)  # pointing left on image
    else:
        up[0] = (thumb_tip > thumb_ip) and (thumb_ip > thumb_mcp)  # pointing right on image

    # Other fingers: tip (y) above PIP (y) => up (remember: y grows downward in image)
    for i, (tip_id, pip_id) in enumerate(zip(TIP_IDS[1:], PIP_IDS[1:]), start=1):
        up[i] = lm.landmark[tip_id].y < lm.landmark[pip_id].y
    return up


def pinch_distance(hand_landmarks, img_w, img_h):
    ix = hand_landmarks.landmark[8].x * img_w
    iy = hand_landmarks.landmark[8].y * img_h
    tx = hand_landmarks.landmark[4].x * img_w
    ty = hand_landmarks.landmark[4].y * img_h
    return math.hypot(ix - tx, iy - ty)


def landmark_xy(hand_landmarks, idx, img_w, img_h):
    l = hand_landmarks.landmark[idx]
    return int(l.x * img_w), int(l.y * img_h)


# -------------------- Drawing state --------------------
class Stroke:
    def __init__(self, color, thickness):
        self.points = []  # list of (x, y)
        self.color = color
        self.thickness = thickness

    def add(self, p):
        self.points.append(tuple(p))

    def draw(self, canvas):
        for i in range(1, len(self.points)):
            cv2.line(canvas, self.points[i-1], self.points[i], self.color, self.thickness, cv2.LINE_AA)


class Shape:
    def __init__(self, kind, p1, p2, color, thickness):
        self.kind = kind  # 'circle', 'rect', 'line'
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.color = color
        self.thickness = thickness

    def draw(self, canvas):
        if self.kind == 'circle':
            r = int(math.hypot(self.p2[0]-self.p1[0], self.p2[1]-self.p1[1]))
            cv2.circle(canvas, self.p1, r, self.color, self.thickness, cv2.LINE_AA)
        elif self.kind == 'rect':
            cv2.rectangle(canvas, self.p1, self.p2, self.color, self.thickness, cv2.LINE_AA)
        elif self.kind == 'line':
            cv2.line(canvas, self.p1, self.p2, self.color, self.thickness, cv2.LINE_AA)


class Label:
    def __init__(self, text, org, color, scale=1.0, thickness=2):
        self.text = text
        self.org = tuple(org)
        self.color = color
        self.scale = scale
        self.thickness = thickness

    def draw(self, canvas):
        cv2.putText(canvas, self.text, self.org, cv2.FONT_HERSHEY_SIMPLEX,
                    self.scale, self.color, self.thickness, cv2.LINE_AA)


class History:
    def __init__(self):
        self.items = []  # each item is Stroke | Shape | Label

    def add(self, item):
        self.items.append(item)

    def undo(self):
        if self.items:
            self.items.pop()

    def redraw(self, canvas):
        h, w = canvas.shape[:2]
        canvas[:] = 0
        for it in self.items:
            it.draw(canvas)


# -------------------- Main App --------------------
class HandDrawApp:
    def __init__(self, cam_index=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.history = History()

        self.color = (255, 255, 255)
        self.thickness = 8

        self.text_mode = False
        self.text_buffer = ""
        self.text_pos = (self.width//2, self.height//2)

        self.preview_shape = None  # Shape being previewed (line/rect/circle)
        self.preview_label = None

        self.curr_stroke = None
        self.last_draw_pos = None

        self.help_on = True

        self.fps = FPS()

        # Slightly lower thresholds so it is less brittle on some cameras
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        self.state = {
            'erase': False,
            'pinching': False,
            'line_anchor': None,  # for line preview
            'rect_anchor': None,
            'circle_anchor': None,
        }

        ensure_dir('saved_images')

    # --------------- gesture helpers ---------------
    def interpret_gestures(self, results, frame):
        """Interpret gestures from detected hands. Returns:
           - draw_point: (x,y) or None (index tip if any hand detected)
           - text_toggle: True/False (if text toggle detected this frame)
           Side-effects: updates self.state and self.preview_shape
        """

        draw_point = None
        text_toggle = False

        img_h, img_w = frame.shape[:2]

        # No hands -> reset transient anchors and preview
        if not results or not results.multi_hand_landmarks:
            # reset transient states
            self.state['pinching'] = False
            self.preview_shape = None
            self.state['line_anchor'] = None
            self.state['rect_anchor'] = None
            self.state['circle_anchor'] = None
            self.state['erase'] = False
            return None, False

        # We'll use the last detected hand's index tip for drawing/erasing/placing text
        # (if you want to favor right-hand only, add logic to choose)
        for hlm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = handed.classification[0].label  # 'Left' or 'Right'
            ups = fingers_up(hlm, handedness)

            # Key points
            index_tip = landmark_xy(hlm, 8, img_w, img_h)

            # Always set draw_point to the index tip of the *last processed* hand
            draw_point = index_tip

            # Pinch detection (px distance between index tip and thumb tip)
            pdist = pinch_distance(hlm, img_w, img_h)
            pinching = pdist < 40  # px threshold; tune per camera

            # Modes
            is_open_palm = ups[1] and ups[2] and ups[3] and ups[4]
            is_index_only = (ups[1] and not ups[2] and not ups[3] and not ups[4])
            is_two_fingers = (ups[1] and ups[2] and not ups[3] and not ups[4])
            is_three_fingers = (ups[1] and ups[2] and ups[3] and not ups[4])
            is_fist = (not ups[1] and not ups[2] and not ups[3] and not ups[4])

            # Erase with open palm (note: draw_point will contain index tip for erasing cursor)
            self.state['erase'] = is_open_palm

            # Text toggle on ✌️
            if is_two_fingers:
                text_toggle = True

            # Draw point remains set above; actual drawing only happens in run() when index-only & not pinch
            # Line preview: pinch to anchor/start; release to place
            if pinching and not self.state['pinching']:
                # newly started pinch
                self.state['line_anchor'] = index_tip
                self.preview_shape = Shape('line', index_tip, index_tip, self.color, self.thickness)
            elif pinching and self.state.get('line_anchor') is not None:
                if self.preview_shape:
                    self.preview_shape.p2 = index_tip
            elif (not pinching) and self.state['pinching'] and self.state.get('line_anchor') is not None:
                # pinch released — place line
                placed = Shape('line', self.state['line_anchor'], index_tip, self.color, self.thickness)
                self.history.add(placed)
                self.preview_shape = None
                self.state['line_anchor'] = None

            # Rectangle: three fingers up then pinch to confirm
            if is_three_fingers and self.state['rect_anchor'] is None:
                self.state['rect_anchor'] = index_tip
                self.preview_shape = Shape('rect', index_tip, index_tip, self.color, self.thickness)
            if self.state['rect_anchor'] is not None and is_three_fingers:
                if self.preview_shape:
                    self.preview_shape.p2 = index_tip
            if self.state['rect_anchor'] is not None and pinching:
                self.history.add(Shape('rect', self.state['rect_anchor'], index_tip, self.color, self.thickness))
                self.preview_shape = None
                self.state['rect_anchor'] = None

            # Circle: fist then extend index to confirm
            if is_fist and self.state['circle_anchor'] is None:
                self.state['circle_anchor'] = index_tip
                self.preview_shape = Shape('circle', index_tip, index_tip, self.color, self.thickness)
            if self.state['circle_anchor'] is not None and is_fist:
                if self.preview_shape:
                    self.preview_shape.p2 = index_tip
            if self.state['circle_anchor'] is not None and ups[1]:  # index extended to confirm
                self.history.add(Shape('circle', self.state['circle_anchor'], index_tip, self.color, self.thickness))
                self.preview_shape = None
                self.state['circle_anchor'] = None

            # update pinching state for next loop
            self.state['pinching'] = pinching

        return draw_point, text_toggle

    # --------------- main loop ---------------
    def run(self):
        if not self.cap.isOpened():
            raise SystemExit("Could not open webcam. Try a different index or grant camera permission.")

        while True:
            ok, frame = self.cap.read()
            if not ok:
                # Don't quit immediately; try again (fixes sudden camera read failures)
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # mirror view

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            draw_pt, text_toggle = self.interpret_gestures(results, frame)
            if text_toggle:
                # toggle only once per detected toggle - simple heuristic: toggle every time interpret reports it
                self.text_mode = not self.text_mode
                self.text_buffer = "" if self.text_mode else self.text_buffer

            # Erase mode draws black circles to remove pixels
            # IMPORTANT fix: draw_pt is now the index tip when any hand exists (so open palm gives an index cursor)
            if self.state['erase'] and draw_pt is not None:
                # Use a slightly larger radius for erase; do not add to history (erase is immediate)
                radius = max(20, int(self.thickness * 2))
                cv2.circle(self.canvas, draw_pt, radius, (0, 0, 0), -1, cv2.LINE_AA)
                self.last_draw_pos = None
                self.curr_stroke = None

            # Normal drawing (index-only) — only add strokes when index-only & not pinching & not erasing & not text_mode
            elif draw_pt is not None and not self.text_mode:
                # Determine if index-only (we need to inspect returned landmarks to be sure).
                # Since interpret_gestures doesn't return finger-up flags directly, a simple heuristic:
                # If preview_shape is None and we are not pinching and not erasing, assume drawing when index moves.
                # To be stricter, we could refactor interpret_gestures to also return finger flags.
                if not self.state['erase'] and not self.state['pinching']:
                    if self.curr_stroke is None:
                        self.curr_stroke = Stroke(self.color, self.thickness)
                        self.history.add(self.curr_stroke)
                    if self.last_draw_pos is None:
                        self.last_draw_pos = draw_pt
                    self.curr_stroke.add(draw_pt)
                    cv2.line(self.canvas, self.last_draw_pos, draw_pt, self.color, self.thickness, cv2.LINE_AA)
                    self.last_draw_pos = draw_pt
                else:
                    # not allowed to draw (either erase or pinching); reset stroke
                    self.curr_stroke = None
                    self.last_draw_pos = None
            else:
                # not drawing
                self.curr_stroke = None
                self.last_draw_pos = None

            # Preview shape overlay
            preview = frame.copy()
            if self.preview_shape is not None:
                self.preview_shape.draw(preview)

            # Text mode: show caret and buffer; press Enter to place
            if self.text_mode:
                # caret at index finger if available else keep last text_pos
                if draw_pt is not None:
                    self.text_pos = draw_pt
                # draw buffer preview
                cv2.putText(preview, self.text_buffer + "_", self.text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color, 2, cv2.LINE_AA)

            # Compose final view: blend canvas over frame
            blended = cv2.addWeighted(preview, 0.5, self.canvas, 1.0, 0)

            # Draw hand landmarks for reference (optional)
            if results and results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(blended, hlm, mp_hands.HAND_CONNECTIONS)

            # HUD
            self.draw_hud(blended)

            cv2.imshow('Hand Gesture Drawing', blended)

            key = cv2.waitKey(1) & 0xFF
            if self.handle_key(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # --------------- UI / HUD ---------------
    def draw_hud(self, img):
        fps = self.fps.tick()
        cv2.putText(img, f"FPS: {fps:4.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Brush: {self.thickness}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2, cv2.LINE_AA)
        mode = 'TEXT' if self.text_mode else ('ERASE' if self.state['erase'] else 'DRAW')
        cv2.putText(img, f"Mode: {mode}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
        if self.help_on:
            y = 100
            help_lines = [
                "[Gestures] Index=Draw | Palm=Erase | ✌︎=Toggle Text | Fist→Index=Circle | Three fingers→Pinch=Rect | Pinch=Line",
                "[Keys] q:Quit s:Save c:Clear z:Undo [ ]:Brush  1/2/3/4:Color  t:Toggle Text h:Toggle Help",
            ]
            for line in help_lines:
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 2, cv2.LINE_AA)
                y += 24

    # --------------- keyboard handling ---------------
    def handle_key(self, key):
        if key == 255:
            return False  # no key
        # print(key)  # debug
        if key in (ord('q'), 27):  # q or ESC
            return True
        if key == ord('h'):
            self.help_on = not self.help_on
        if key == ord('c'):
            self.canvas[:] = 0
            self.history = History()
        if key == ord('z'):
            self.history.undo()
            self.history.redraw(self.canvas)
        if key == ord('s'):
            ensure_dir('saved_images')
            name = timestamp_name()
            path = os.path.join('saved_images', name)
            cv2.imwrite(path, self.canvas)
            print(f"Saved: {path}")
        if key == ord('['):
            self.thickness = max(1, self.thickness - 1)
        if key == ord(']'):
            self.thickness = min(50, self.thickness + 1)
        if key == ord('1'):
            self.color = (255,255,255)
        if key == ord('2'):
            self.color = (0,0,255)  # red (BGR)
        if key == ord('3'):
            self.color = (0,255,0)  # green
        if key == ord('4'):
            self.color = (255,0,0)  # blue
        if key == ord('t'):
            self.text_mode = not self.text_mode
            if self.text_mode:
                self.text_buffer = ""
        # Text typing (only when text_mode)
        if self.text_mode:
            if key == 13 or key == 10:  # Enter
                if self.text_buffer.strip():
                    self.history.add(Label(self.text_buffer, self.text_pos, self.color, scale=1.0, thickness=2))
                    self.history.redraw(self.canvas)
                    self.text_buffer = ""
                    self.text_mode = False
            elif key in (8, 127):  # Backspace/Delete
                self.text_buffer = self.text_buffer[:-1]
            else:
                try:
                    ch = chr(key)
                except ValueError:
                    ch = ''
                if ch and ch.isprintable():
                    self.text_buffer += ch
        return False


if __name__ == '__main__':
    HandDrawApp().run()
