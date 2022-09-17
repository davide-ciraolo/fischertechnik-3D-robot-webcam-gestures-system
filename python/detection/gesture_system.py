import math
import time
import cv2
import mediapipe as mp


class GestureSystem:

    def __init__(self):
        self._NO_HAND = "no_hand"
        self._GESTURE_NONE = "none"
        self._GESTURE_STOP = "stop"
        self._GESTURE_FORWARD = "forward"
        self._GESTURE_BACKWARD = "backward"
        self._GESTURE_UP = "up"
        self._GESTURE_DOWN = "down"
        self._GESTURE_RIGHT = "right"
        self._GESTURE_LEFT = "left"
        self._GESTURE_OPEN = "open"
        self._GESTURE_CLOSE = "close"
        self._detecting = False
        self._current_gesture_counter = 0
        self._last_gesture = self._NO_HAND
        self._current_gesture = self._NO_HAND
        self._detected_gesture = self._NO_HAND
        self._gesture_changed = True
        self._last_gesture_time = 0

    def start_detection(self, callback):
        self._detecting = True
        self.detect(callback)

    def stop_detection(self):
        self._detecting = False

    @staticmethod
    def get_relevant_dots(image, landmarks):
        d0 = (landmarks[0][1], landmarks[0][2])  # wrist
        cv2.circle(image, (d0[0], d0[1]), 15, (255, 255, 255), 3)

        d5 = (landmarks[4][1], landmarks[4][2])  # thumb
        cv2.circle(image, (d5[0], d5[1]), 15, (255, 255, 255), 3)

        d4 = (landmarks[8][1], landmarks[8][2])  # index
        cv2.circle(image, (d4[0], d4[1]), 15, (255, 255, 255), 3)

        d3 = (landmarks[12][1], landmarks[12][2])  # middle
        cv2.circle(image, (d3[0], d3[1]), 15, (255, 255, 255), 3)

        d2 = (landmarks[16][1], landmarks[16][2])  # annular
        cv2.circle(image, (d2[0], d2[1]), 15, (255, 255, 255), 3)

        d1 = (landmarks[20][1], landmarks[20][2])  # pinkie
        cv2.circle(image, (d1[0], d1[1]), 15, (255, 255, 255), 3)

        return d0, d1, d2, d3, d4, d5

    @staticmethod
    def forward_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.5 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.2
        wrist_distance_d2 = 0.5 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.2
        wrist_distance_d3 = 0.5 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.2
        wrist_distance_d5 = 0.5 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 1.2

        d1_d4_check = math.hypot(d1[0] - d4[0], d1[1] - d4[1]) / reference > 0.25
        d5_d4_check = math.hypot(d5[0] - d4[0], d5[1] - d4[1]) / reference > 0.4
        d5_check = d5[0] - d0[0] > 0

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5 \
            and d1_d4_check and d5_d4_check and d5_check

    @staticmethod
    def backward_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.5 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.2
        wrist_distance_d2 = 0.5 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.2
        wrist_distance_d3 = 0.5 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.2
        wrist_distance_d5 = 0.5 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 1.2

        d5_d4_check = math.hypot(d5[0] - d4[0], d5[1] - d4[1]) / reference > 0.4
        d5_d1_check = math.hypot(d5[0] - d1[0], d5[1] - d1[1]) / reference > 0.4
        d1_d4_check = math.hypot(d1[0] - d4[0], d1[1] - d4[1]) / reference > 0.25
        d5_check = d5[0] - d0[0] < 0

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5 \
            and d5_d4_check and d1_d4_check and d5_d1_check and d5_check

    @staticmethod
    def stop_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.6 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.3
        wrist_distance_d2 = 0.6 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.3
        wrist_distance_d3 = 0.6 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.3
        wrist_distance_d5 = 0.8 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 1.3

        d1_d4_check = math.hypot(d1[0] - d4[0], d1[1] - d4[1]) / reference > 0.2
        d5_d4_check = math.hypot(d5[0] - d4[0], d5[1] - d4[1]) / reference < 0.5
        d5_d1_check = math.hypot(d5[0] - d1[0], d5[1] - d1[1]) / reference < 0.5
        d5_checks = d5_d4_check or d5_d1_check

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5\
            and d1_d4_check and d5_checks

    @staticmethod
    def open_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.8 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.3
        wrist_distance_d2 = 0.8 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.3
        wrist_distance_d3 = 0.8 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.3
        wrist_distance_d5 = math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 1.3

        d1_d4_check = math.hypot(d1[0] - d4[0], d1[1] - d4[1]) / reference < 0.25
        d5_d4_check = math.hypot(d5[0] - d4[0], d5[1] - d4[1]) / reference > 0.2
        d5_d1_check = math.hypot(d5[0] - d1[0], d5[1] - d1[1]) / reference > 0.2
        d5_checks = d5_d4_check and d5_d1_check

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5\
            and d1_d4_check and d5_checks

    @staticmethod
    def close_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.8 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.3
        wrist_distance_d2 = 0.8 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.3
        wrist_distance_d3 = 0.8 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.3
        wrist_distance_d5 = math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 1.3

        d1_d4_check = math.hypot(d1[0] - d4[0], d1[1] - d4[1]) / reference < 0.25
        d5_d4_check = math.hypot(d5[0] - d4[0], d5[1] - d4[1]) / reference <= 0.2
        d5_d1_check = math.hypot(d5[0] - d1[0], d5[1] - d1[1]) / reference <= 0.2
        d5_checks = d5_d4_check and d5_d1_check

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5\
            and d1_d4_check and d5_checks

    @staticmethod
    def left_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.6 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.3
        wrist_distance_d2 = 0.6 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.3
        wrist_distance_d3 = 0.6 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.3
        wrist_distance_d5 = 1.5 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference

        d5_check = d5[0] - d0[0] > 0

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5 and d5_check

    @staticmethod
    def right_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.6 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 1.3
        wrist_distance_d2 = 0.6 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 1.3
        wrist_distance_d3 = 0.6 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.3
        wrist_distance_d5 = 1.5 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference

        d5_check = d5[0] - d0[0] < 0

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5 and d5_check

    @staticmethod
    def up_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.2 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 0.8
        wrist_distance_d2 = 0.2 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 0.8
        wrist_distance_d3 = 0.2 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 0.8
        wrist_distance_d5 = 0.2 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 0.8

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5

    @staticmethod
    def down_detect(d0, d1, d2, d3, d4, d5):
        reference = math.hypot(d4[0] - d0[0], d4[1] - d0[1])  # distance between wrist and index

        wrist_distance_d1 = 0.2 < math.hypot(d1[0] - d0[0], d1[1] - d0[1]) / reference < 0.8
        wrist_distance_d2 = 0.2 < math.hypot(d2[0] - d0[0], d2[1] - d0[1]) / reference < 0.8
        wrist_distance_d3 = 0.9 < math.hypot(d3[0] - d0[0], d3[1] - d0[1]) / reference < 1.5
        wrist_distance_d5 = 0.2 < math.hypot(d5[0] - d0[0], d5[1] - d0[1]) / reference < 0.8

        return wrist_distance_d1 and wrist_distance_d2 and wrist_distance_d3 and wrist_distance_d5

    def gesture_detect(self, d0, d1, d2, d3, d4, d5):

        if GestureSystem.up_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_UP
        elif GestureSystem.down_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_DOWN
        elif GestureSystem.right_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_RIGHT
        elif GestureSystem.left_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_LEFT
        elif GestureSystem.stop_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_STOP
        elif GestureSystem.open_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_OPEN
        elif GestureSystem.close_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_CLOSE
        elif GestureSystem.forward_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_FORWARD
        elif GestureSystem.backward_detect(d0, d1, d2, d3, d4, d5):
            self._current_gesture = self._GESTURE_BACKWARD
        else:
            self._current_gesture = self._GESTURE_NONE

        if self._last_gesture == self._current_gesture:
            self._current_gesture_counter += 1
        else:
            self._current_gesture_counter = 0

        self._last_gesture = self._current_gesture

        if self._current_gesture_counter >= 5:
            if self._detected_gesture != self._current_gesture:
                self._detected_gesture = self._current_gesture
                self._gesture_changed = True
            self._current_gesture_counter = 0

        return self._detected_gesture

    def detect(self, callback):
        hands = mp.solutions.hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

        camera = cv2.VideoCapture(0)

        while self._detecting:
            success, img = camera.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            hand_detected = results.multi_hand_landmarks is not None
            landmarks = []
            if hand_detected:
                for hand_lms in results.multi_hand_landmarks:
                    for id_lm, lm in enumerate(hand_lms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks.append([id_lm, cx, cy])
                        cv2.circle(img, (cx, cy), 3, (255, 0, 255), 3)
                    mp.solutions.drawing_utils.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
                d0, d1, d2, d3, d4, d5 = GestureSystem.get_relevant_dots(img, landmarks)
                self.gesture_detect(d0, d1, d2, d3, d4, d5)
            else:
                if self._detected_gesture != self._NO_HAND:
                    self._detected_gesture = self._NO_HAND
                    self._gesture_changed = True

            if self._gesture_changed or ((time.time() - self._last_gesture_time) > 2):
                callback(self._detected_gesture)
                self._gesture_changed = False
                self._last_gesture_time = time.time()

            cv2.putText(img, str(self._detected_gesture), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(100)

        camera.release()
        cv2.destroyAllWindows()
