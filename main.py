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

PINCH_THRESHOLD = 0.35
RELEASE_THRESHOLD = 0.70

# A pinch shorter than this becomes a normal left click
DRAG_HOLD_TIME = 0.65

# Prevent tiny accidental pinches
MIN_CLICK_TIME = 0.08

# Right-click confirmation
RIGHT_CONFIRM_FRAMES = 4

# Right-click release confirmation
RIGHT_RELEASE_FRAMES = 5


# =========================================================
# 6. LEFT CLICK / DRAG STATE
# =========================================================

left_pinching = False

left_pinch_start_time = 0

dragging = False


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
# 9. WEBCAM SETUP
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam.")

    landmarker.close()

    raise SystemExit


print()
print("==============================================")
print("           SMART VIRTUAL MOUSE")
print("==============================================")
print("Index finger            -> Move cursor")
print("Quick Thumb + Index     -> LEFT CLICK")
print("Hold Thumb + Index      -> START DRAG")
print("Move while holding      -> DRAG")
print("Release Thumb + Index   -> DROP")
print("Thumb + Middle          -> RIGHT CLICK")
print("Q                       -> Quit")
print("==============================================")
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
    # ACTIVE AREA
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
        # Convert landmarks to pixels
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


        # Index fingertip
        cv2.circle(
            frame,
            index_tip,
            10,
            (255, 0, 0),
            -1
        )


        # Thumb fingertip
        cv2.circle(
            frame,
            thumb_tip,
            10,
            (0, 255, 255),
            -1
        )


        # Middle fingertip
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
                (
                    target_x
                    -
                    previous_x
                )
                /
                SMOOTHENING
            )


            current_y = (
                previous_y
                +
                (
                    target_y
                    -
                    previous_y
                )
                /
                SMOOTHENING
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
        # 12. NORMALIZED DISTANCES
        # =================================================

        hand_size = math.dist(
            index_base,
            pinky_base
        )


        if hand_size > 0:

            left_distance = (
                math.dist(
                    thumb_tip,
                    index_tip
                )
                /
                hand_size
            )


            right_distance = (
                math.dist(
                    thumb_tip,
                    middle_tip
                )
                /
                hand_size
            )

        else:

            left_distance = 999
            right_distance = 999


        # =================================================
        # GESTURE LINES
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
        # 13. LEFT CLICK / DRAG STATE MACHINE
        # =================================================

        current_time = time.monotonic()


        # -------------------------------------------------
        # PINCH START
        # -------------------------------------------------

        if (
            left_distance < PINCH_THRESHOLD
            and
            not left_pinching
        ):

            left_pinching = True

            left_pinch_start_time = current_time

            print("LEFT PINCH START")


        # -------------------------------------------------
        # PINCH HELD
        # -------------------------------------------------

        if (
            left_pinching
            and
            left_distance < PINCH_THRESHOLD
        ):

            pinch_duration = (
                current_time
                -
                left_pinch_start_time
            )


            # ---------------------------------------------
            # Convert long pinch into drag
            # ---------------------------------------------

            if (
                pinch_duration >= DRAG_HOLD_TIME
                and
                not dragging
            ):

                try:

                    pyautogui.mouseDown(
                        button="left"
                    )

                    dragging = True

                    print("DRAG START")

                    action_message = "DRAG START"

                    action_message_until = (
                        current_time + 0.7
                    )

                except pyautogui.FailSafeException:

                    pass


        # -------------------------------------------------
        # PINCH RELEASED
        # -------------------------------------------------

        if (
            left_pinching
            and
            left_distance > RELEASE_THRESHOLD
        ):

            pinch_duration = (
                current_time
                -
                left_pinch_start_time
            )


            # ---------------------------------------------
            # If dragging -> DROP
            # ---------------------------------------------

            if dragging:

                try:

                    pyautogui.mouseUp(
                        button="left"
                    )

                except pyautogui.FailSafeException:

                    pass


                dragging = False

                print("DROP")

                action_message = "DROP!"

                action_message_until = (
                    current_time + 0.7
                )


            # ---------------------------------------------
            # Otherwise -> LEFT CLICK
            # ---------------------------------------------

            elif (
                pinch_duration
                >= MIN_CLICK_TIME
                and
                pinch_duration
                < DRAG_HOLD_TIME
            ):

                try:

                    pyautogui.click(
                        button="left"
                    )

                    print("LEFT CLICK")

                    action_message = "LEFT CLICK!"

                    action_message_until = (
                        current_time + 0.5
                    )

                except pyautogui.FailSafeException:

                    pass


            # Reset pinch
            left_pinching = False

            left_pinch_start_time = 0


        # =================================================
        # 14. RIGHT CLICK
        # =================================================

        # Do not process right click while dragging
        if not dragging:

            if right_distance < PINCH_THRESHOLD:

                right_pinch_frames += 1

                right_release_frames = 0


            elif right_distance > RELEASE_THRESHOLD:

                right_pinch_frames = 0


                if right_locked:

                    right_release_frames += 1


                    if (
                        right_release_frames
                        >= RIGHT_RELEASE_FRAMES
                    ):

                        right_locked = False

                        right_release_frames = 0

                        print(
                            "RIGHT CLICK READY"
                        )


                else:

                    right_release_frames = 0


            else:

                right_pinch_frames = 0

                right_release_frames = 0


            # ---------------------------------------------
            # Execute right click
            # ---------------------------------------------

            if (
                right_pinch_frames
                >= RIGHT_CONFIRM_FRAMES
                and
                not right_locked
                and
                left_distance
                > PINCH_THRESHOLD
            ):

                try:

                    pyautogui.click(
                        button="right"
                    )

                    print("RIGHT CLICK")

                    action_message = (
                        "RIGHT CLICK!"
                    )

                    action_message_until = (
                        current_time + 0.5
                    )

                except pyautogui.FailSafeException:

                    pass


                right_locked = True

                right_pinch_frames = 0

                right_release_frames = 0


        # =================================================
        # 15. DISPLAY INFORMATION
        # =================================================

        cv2.putText(
            frame,
            f"Left Pinch: {left_distance:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Right Pinch: {right_distance:.2f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        # -------------------------------------------------
        # Current mouse state
        # -------------------------------------------------

        if dragging:

            mouse_state = "DRAGGING"

        elif left_pinching:

            mouse_state = "PINCHING"

        else:

            mouse_state = "READY"


        cv2.putText(
            frame,
            f"Mouse State: {mouse_state}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


        # -------------------------------------------------
        # Pinch timer
        # -------------------------------------------------

        if left_pinching:

            hold_time = (
                current_time
                -
                left_pinch_start_time
            )


            cv2.putText(
                frame,
                f"Hold: {hold_time:.2f}s",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )


        # -------------------------------------------------
        # Drag indicator
        # -------------------------------------------------

        if dragging:

            cv2.putText(
                frame,
                "DRAGGING",
                (
                    width // 2 - 80,
                    height - 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )


    # =====================================================
    # 16. HAND LOST
    # =====================================================

    else:

        right_pinch_frames = 0
        right_release_frames = 0


        # Safety:
        # if hand disappears during drag,
        # release mouse button.
        if dragging:

            try:

                pyautogui.mouseUp(
                    button="left"
                )

            except pyautogui.FailSafeException:

                pass


            dragging = False
            left_pinching = False

            print(
                "DRAG CANCELLED - HAND LOST"
            )


    # =====================================================
    # 17. ACTION MESSAGE
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
    # 18. DISPLAY
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
# 19. CLEANUP
# =========================================================

# Important:
# never leave mouse button held when program closes.
if dragging:

    try:

        pyautogui.mouseUp(
            button="left"
        )

    except pyautogui.FailSafeException:

        pass


cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("Program closed.")