import cv2
import mediapipe as mp
import time
import pyautogui
import math


# =========================================================
# 1. MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# =========================================================
# 2. HAND CONNECTIONS
# =========================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


# =========================================================
# 3. SCREEN / MOUSE SETTINGS
# =========================================================

screen_width, screen_height = pyautogui.size()

print(
    f"Screen Resolution: "
    f"{screen_width} x {screen_height}"
)

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True


# =========================================================
# 4. CURSOR SETTINGS
# =========================================================

SMOOTHENING = 7
FRAME_MARGIN = 100

previous_x = screen_width / 2
previous_y = screen_height / 2


# =========================================================
# 5. GESTURE SETTINGS
# =========================================================

# Number of stable frames needed before action
GESTURE_CONFIRM_FRAMES = 4

# Pinch threshold
PINCH_THRESHOLD = 0.35

# Finger must move beyond this value before unlocking
RELEASE_THRESHOLD = 0.70

# Stable release frames
RELEASE_CONFIRM_FRAMES = 5


# =========================================================
# 6. LEFT CLICK STATE
# =========================================================

left_pinch_frames = 0
left_release_frames = 0
left_locked = False


# =========================================================
# 7. RIGHT CLICK STATE
# =========================================================

right_pinch_frames = 0
right_release_frames = 0
right_locked = False


# =========================================================
# 8. VISUAL MESSAGE
# =========================================================

action_message = ""
action_message_until = 0


# =========================================================
# 9. WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam.")

    landmarker.close()

    raise SystemExit


print()
print("==========================================")
print("         SMART VIRTUAL MOUSE")
print("==========================================")
print("Index finger          -> Move cursor")
print("Thumb + Index pinch   -> LEFT CLICK")
print("Thumb + Middle pinch  -> RIGHT CLICK")
print("Q                     -> Quit")
print("==========================================")
print()


start_time = time.monotonic()


# =========================================================
# 10. MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # =====================================================
    # ACTIVE CONTROL AREA
    # =====================================================

    cv2.rectangle(
        frame,
        (FRAME_MARGIN, FRAME_MARGIN),
        (
            width - FRAME_MARGIN,
            height - FRAME_MARGIN
        ),
        (255, 255, 0),
        2
    )


    # =====================================================
    # MEDIAPIPE IMAGE
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # =====================================================
    # HAND FOUND
    # =====================================================

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        points = []


        # -------------------------------------------------
        # Landmark -> pixel coordinates
        # -------------------------------------------------

        for landmark in hand_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )


        # -------------------------------------------------
        # Draw hand skeleton
        # -------------------------------------------------

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )


        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                (0, 0, 255),
                -1
            )


        # =================================================
        # IMPORTANT LANDMARKS
        # =================================================

        thumb_tip = points[4]

        index_base = points[5]
        index_tip = points[8]

        middle_tip = points[12]

        pinky_base = points[17]


        finger_x, finger_y = index_tip


        # -------------------------------------------------
        # Highlight important fingertips
        # -------------------------------------------------

        # Index = blue
        cv2.circle(
            frame,
            index_tip,
            10,
            (255, 0, 0),
            -1
        )

        # Thumb = yellow
        cv2.circle(
            frame,
            thumb_tip,
            10,
            (0, 255, 255),
            -1
        )

        # Middle = orange-ish
        cv2.circle(
            frame,
            middle_tip,
            10,
            (0, 165, 255),
            -1
        )


        # =================================================
        # 11. CURSOR MOVEMENT
        # =================================================

        if (
            FRAME_MARGIN < finger_x
            < width - FRAME_MARGIN
            and
            FRAME_MARGIN < finger_y
            < height - FRAME_MARGIN
        ):

            target_x = (
                (finger_x - FRAME_MARGIN)
                /
                (width - 2 * FRAME_MARGIN)
            ) * screen_width


            target_y = (
                (finger_y - FRAME_MARGIN)
                /
                (height - 2 * FRAME_MARGIN)
            ) * screen_height


            target_x = max(
                1,
                min(
                    screen_width - 2,
                    target_x
                )
            )

            target_y = max(
                1,
                min(
                    screen_height - 2,
                    target_y
                )
            )


            current_x = (
                previous_x
                +
                (target_x - previous_x)
                / SMOOTHENING
            )

            current_y = (
                previous_y
                +
                (target_y - previous_y)
                / SMOOTHENING
            )


            try:

                pyautogui.moveTo(
                    current_x,
                    current_y
                )

            except pyautogui.FailSafeException:

                pass


            previous_x = current_x
            previous_y = current_y


        # =================================================
        # 12. NORMALIZATION
        # =================================================

        hand_size = math.dist(
            index_base,
            pinky_base
        )


        if hand_size > 0:

            # Thumb ↔ Index
            left_distance = (
                math.dist(
                    thumb_tip,
                    index_tip
                )
                / hand_size
            )


            # Thumb ↔ Middle
            right_distance = (
                math.dist(
                    thumb_tip,
                    middle_tip
                )
                / hand_size
            )

        else:

            left_distance = 999
            right_distance = 999


        # =================================================
        # DRAW GESTURE LINES
        # =================================================

        # Thumb -> Index
        cv2.line(
            frame,
            thumb_tip,
            index_tip,
            (255, 0, 255),
            3
        )


        # Thumb -> Middle
        cv2.line(
            frame,
            thumb_tip,
            middle_tip,
            (0, 255, 255),
            2
        )


        # =================================================
        # 13. LEFT CLICK DETECTION
        # =================================================

        if left_distance < PINCH_THRESHOLD:

            left_pinch_frames += 1
            left_release_frames = 0

        elif left_distance > RELEASE_THRESHOLD:

            left_pinch_frames = 0

            if left_locked:

                left_release_frames += 1

                if (
                    left_release_frames
                    >= RELEASE_CONFIRM_FRAMES
                ):

                    left_locked = False
                    left_release_frames = 0

                    print("LEFT CLICK READY")

            else:

                left_release_frames = 0

        else:

            left_pinch_frames = 0
            left_release_frames = 0


        # =================================================
        # 14. RIGHT CLICK DETECTION
        # =================================================

        if right_distance < PINCH_THRESHOLD:

            right_pinch_frames += 1
            right_release_frames = 0

        elif right_distance > RELEASE_THRESHOLD:

            right_pinch_frames = 0

            if right_locked:

                right_release_frames += 1

                if (
                    right_release_frames
                    >= RELEASE_CONFIRM_FRAMES
                ):

                    right_locked = False
                    right_release_frames = 0

                    print("RIGHT CLICK READY")

            else:

                right_release_frames = 0

        else:

            right_pinch_frames = 0
            right_release_frames = 0


        # =================================================
        # 15. EXECUTE LEFT CLICK
        # =================================================

        if (
            left_pinch_frames
            >= GESTURE_CONFIRM_FRAMES
            and
            not left_locked
        ):

            try:

                pyautogui.click(
                    button="left"
                )

                print("LEFT CLICK")

            except pyautogui.FailSafeException:

                pass


            left_locked = True

            left_pinch_frames = 0
            left_release_frames = 0


            action_message = "LEFT CLICK!"

            action_message_until = (
                time.monotonic() + 0.5
            )


        # =================================================
        # 16. EXECUTE RIGHT CLICK
        # =================================================

        if (
            right_pinch_frames
            >= GESTURE_CONFIRM_FRAMES
            and
            not right_locked
        ):

            # Avoid interpreting an index pinch
            # as a right click at the same time.
            if (
                left_distance
                > PINCH_THRESHOLD
            ):

                try:

                    pyautogui.click(
                        button="right"
                    )

                    print("RIGHT CLICK")

                except pyautogui.FailSafeException:

                    pass


                right_locked = True

                right_pinch_frames = 0
                right_release_frames = 0


                action_message = "RIGHT CLICK!"

                action_message_until = (
                    time.monotonic() + 0.5
                )


        # =================================================
        # 17. SCREEN INFORMATION
        # =================================================

        cv2.putText(
            frame,
            f"Index: {index_tip}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Left Pinch: {left_distance:.2f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Right Pinch: {right_distance:.2f}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        if left_locked:

            left_state = "LOCKED"

        else:

            left_state = "READY"


        if right_locked:

            right_state = "LOCKED"

        else:

            right_state = "READY"


        cv2.putText(
            frame,
            f"Left: {left_state}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Right: {right_state}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # =====================================================
    # NO HAND
    # =====================================================

    else:

        left_pinch_frames = 0
        right_pinch_frames = 0

        left_release_frames = 0
        right_release_frames = 0


    # =====================================================
    # ACTION MESSAGE
    # =====================================================

    if time.monotonic() < action_message_until:

        cv2.putText(
            frame,
            action_message,
            (
                width // 2 - 120,
                60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Smart Virtual Mouse",
        frame
    )


    if (
        cv2.waitKey(1) & 0xFF
        ==
        ord("q")
    ):

        break


# =========================================================
# 18. CLEANUP
# =========================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("Program closed.")