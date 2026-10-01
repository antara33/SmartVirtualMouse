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
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]


# =========================================================
# 3. SCREEN AND MOUSE SETTINGS
# =========================================================

screen_width, screen_height = pyautogui.size()

print(
    f"Screen Resolution: "
    f"{screen_width} x {screen_height}"
)

# Remove PyAutoGUI's built-in delay
pyautogui.PAUSE = 0

# Keep emergency corner fail-safe enabled
pyautogui.FAILSAFE = True


# =========================================================
# 4. CURSOR SETTINGS
# =========================================================

# Higher value = smoother but slightly slower cursor
SMOOTHENING = 7

previous_x = screen_width / 2
previous_y = screen_height / 2

# Mouse-control area inside webcam frame
FRAME_MARGIN = 100


# =========================================================
# 5. LEFT CLICK SETTINGS
# =========================================================

# Pinch must remain valid for several frames before clicking
PINCH_CONFIRM_FRAMES = 4

# Thumb + index must be closer than this
PINCH_THRESHOLD = 0.35

# Fingers must separate beyond this before next click
PINCH_RELEASE_THRESHOLD = 0.70

# Release must also remain stable
RELEASE_CONFIRM_FRAMES = 5

pinch_frame_count = 0
release_frame_count = 0

# Prevent multiple clicks during one pinch
click_locked = False

# Controls temporary "LEFT CLICK!" message
click_message_until = 0


# =========================================================
# 6. WEBCAM SETUP
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam.")

    landmarker.close()

    raise SystemExit


print()
print("========================================")
print("      SMART VIRTUAL MOUSE")
print("========================================")
print("Index finger  -> Move cursor")
print("Thumb + Index -> Left click")
print("Press Q       -> Quit")
print("========================================")
print()


start_time = time.monotonic()


# =========================================================
# 7. MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Error: Could not read webcam frame.")

        break


    # -----------------------------------------------------
    # Mirror webcam
    # -----------------------------------------------------

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # -----------------------------------------------------
    # Draw active mouse-control area
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Convert OpenCV BGR -> RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # Create MediaPipe Image
    # -----------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------------------------------
    # Generate timestamp
    # -----------------------------------------------------

    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )


    # -----------------------------------------------------
    # Detect hand
    # -----------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # =====================================================
    # 8. PROCESS DETECTED HAND
    # =====================================================

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        points = []


        # -------------------------------------------------
        # Convert normalized landmarks -> pixel coordinates
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
        # Draw hand connections
        # -------------------------------------------------

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )


        # -------------------------------------------------
        # Draw all 21 landmarks
        # -------------------------------------------------

        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                (0, 0, 255),
                -1
            )


        # =================================================
        # 9. IMPORTANT LANDMARKS
        # =================================================

        # Thumb tip
        thumb_tip = points[4]

        # Index MCP / base
        index_base = points[5]

        # Index fingertip
        index_tip = points[8]

        # Pinky MCP / base
        pinky_base = points[17]


        finger_x, finger_y = index_tip


        # -------------------------------------------------
        # Highlight index fingertip
        # -------------------------------------------------

        cv2.circle(
            frame,
            index_tip,
            10,
            (255, 0, 0),
            -1
        )


        # -------------------------------------------------
        # Highlight thumb fingertip
        # -------------------------------------------------

        cv2.circle(
            frame,
            thumb_tip,
            10,
            (0, 255, 255),
            -1
        )


        # =================================================
        # 10. CURSOR MOVEMENT
        # =================================================

        if (
            FRAME_MARGIN < finger_x
            < width - FRAME_MARGIN
            and
            FRAME_MARGIN < finger_y
            < height - FRAME_MARGIN
        ):

            # ---------------------------------------------
            # Map camera X -> screen X
            # ---------------------------------------------

            target_x = (
                (finger_x - FRAME_MARGIN)
                /
                (width - 2 * FRAME_MARGIN)
            ) * screen_width


            # ---------------------------------------------
            # Map camera Y -> screen Y
            # ---------------------------------------------

            target_y = (
                (finger_y - FRAME_MARGIN)
                /
                (height - 2 * FRAME_MARGIN)
            ) * screen_height


            # ---------------------------------------------
            # Keep coordinates inside screen
            # ---------------------------------------------

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


            # ---------------------------------------------
            # Cursor smoothing
            # ---------------------------------------------

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


            # ---------------------------------------------
            # Move actual Windows cursor
            # ---------------------------------------------

            try:

                pyautogui.moveTo(
                    current_x,
                    current_y
                )

            except pyautogui.FailSafeException:

                print(
                    "PyAutoGUI fail-safe triggered."
                )


            previous_x = current_x
            previous_y = current_y


            # ---------------------------------------------
            # Display cursor coordinates
            # ---------------------------------------------

            cv2.putText(
                frame,
                (
                    f"Mouse: "
                    f"({int(current_x)}, "
                    f"{int(current_y)})"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        # =================================================
        # 11. CALCULATE PINCH DISTANCE
        # =================================================

        # Raw distance between thumb and index
        pinch_distance = math.dist(
            thumb_tip,
            index_tip
        )


        # Palm width is used as hand-size reference
        hand_size = math.dist(
            index_base,
            pinky_base
        )


        # -------------------------------------------------
        # Normalize pinch distance
        # -------------------------------------------------

        if hand_size > 0:

            normalized_pinch = (
                pinch_distance
                /
                hand_size
            )

        else:

            normalized_pinch = 999


        # -------------------------------------------------
        # Draw line between thumb and index
        # -------------------------------------------------

        cv2.line(
            frame,
            thumb_tip,
            index_tip,
            (255, 0, 255),
            3
        )


        # =================================================
        # 12. PINCH / RELEASE STATE
        # =================================================

        if normalized_pinch < PINCH_THRESHOLD:

            # ---------------------------------------------
            # Possible pinch
            # ---------------------------------------------

            pinch_frame_count += 1

            release_frame_count = 0


            cv2.putText(
                frame,
                "PINCH DETECTED",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


        elif normalized_pinch > PINCH_RELEASE_THRESHOLD:

            # ---------------------------------------------
            # Fingers clearly separated
            # ---------------------------------------------

            pinch_frame_count = 0


            if click_locked:

                release_frame_count += 1


                # -----------------------------------------
                # Confirm release for several frames
                # -----------------------------------------

                if (
                    release_frame_count
                    >= RELEASE_CONFIRM_FRAMES
                ):

                    click_locked = False

                    release_frame_count = 0

                    print("CLICK READY")


            else:

                release_frame_count = 0


        else:

            # ---------------------------------------------
            # DEAD ZONE
            #
            # 0.35 <= pinch <= 0.70
            #
            # This prevents noisy landmark measurements
            # from repeatedly locking/unlocking click.
            # ---------------------------------------------

            pinch_frame_count = 0

            release_frame_count = 0


        # =================================================
        # 13. LEFT CLICK
        # =================================================

        if (
            pinch_frame_count
            >= PINCH_CONFIRM_FRAMES
            and
            not click_locked
        ):

            try:

                pyautogui.click()

                print("LEFT CLICK")

            except pyautogui.FailSafeException:

                print(
                    "Click cancelled by "
                    "PyAutoGUI fail-safe."
                )


            # ---------------------------------------------
            # Lock click
            # ---------------------------------------------

            click_locked = True


            # Reset counters
            pinch_frame_count = 0
            release_frame_count = 0


            # Show feedback for half a second
            click_message_until = (
                time.monotonic() + 0.5
            )


        # =================================================
        # 14. DISPLAY INFORMATION
        # =================================================

        cv2.putText(
            frame,
            f"Index: {index_tip}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Pinch: {normalized_pinch:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # -------------------------------------------------
        # Show current click state
        # -------------------------------------------------

        if click_locked:

            state_text = "Click State: LOCKED"

        else:

            state_text = "Click State: READY"


        cv2.putText(
            frame,
            state_text,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


    # =====================================================
    # 15. NO HAND DETECTED
    # =====================================================

    else:

        # Reset only unfinished gesture counters.
        #
        # Do NOT automatically unlock a completed click
        # here because a momentary hand-detection failure
        # could otherwise produce repeated clicks.

        pinch_frame_count = 0
        release_frame_count = 0


    # =====================================================
    # 16. CLICK VISUAL FEEDBACK
    # =====================================================

    if time.monotonic() < click_message_until:

        cv2.putText(
            frame,
            "LEFT CLICK!",
            (
                width // 2 - 100,
                60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


    # =====================================================
    # 17. DISPLAY WEBCAM
    # =====================================================

    cv2.imshow(
        "Smart Virtual Mouse",
        frame
    )


    # -----------------------------------------------------
    # Press Q to quit
    # -----------------------------------------------------

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