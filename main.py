import cv2
import mediapipe as mp
import time
import pyautogui


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
# 3. MOUSE SETTINGS
# =========================================================

screen_width, screen_height = pyautogui.size()

print(
    f"Screen Resolution: "
    f"{screen_width} x {screen_height}"
)

# Remove PyAutoGUI delay
pyautogui.PAUSE = 0

# Cursor smoothing
smoothening = 7

previous_x = screen_width / 2
previous_y = screen_height / 2

# Webcam control-area margin
frame_margin = 100


# =========================================================
# 4. WEBCAM SETUP
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Smart Virtual Mouse Started!")
print("Move your index finger inside the blue box.")
print("Press Q to close.")

start_time = time.monotonic()


# =========================================================
# 5. MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # -----------------------------------------------------
    # Draw mouse-control area
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (frame_margin, frame_margin),
        (
            width - frame_margin,
            height - frame_margin
        ),
        (255, 255, 0),
        2
    )


    # -----------------------------------------------------
    # Convert BGR -> RGB
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
    # Timestamp for VIDEO mode
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
    # 6. PROCESS DETECTED HAND
    # =====================================================

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            points = []

            # ---------------------------------------------
            # Convert normalized landmarks to pixels
            # ---------------------------------------------

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))


            # ---------------------------------------------
            # Draw hand connections
            # ---------------------------------------------

            for start, end in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )


            # ---------------------------------------------
            # Draw all 21 landmarks
            # ---------------------------------------------

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 0, 255),
                    -1
                )


            # =============================================
            # INDEX FINGERTIP
            # Landmark 8
            # =============================================

            index_tip = points[8]

            finger_x, finger_y = index_tip


            # Highlight index fingertip
            cv2.circle(
                frame,
                index_tip,
                10,
                (255, 0, 0),
                -1
            )


            # Display camera coordinates
            cv2.putText(
                frame,
                f"Index: {index_tip}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            # =============================================
            # 7. VIRTUAL MOUSE MOVEMENT
            # =============================================

            if (
                frame_margin < finger_x
                < width - frame_margin
                and
                frame_margin < finger_y
                < height - frame_margin
            ):

                # -----------------------------------------
                # Map camera X to screen X
                # -----------------------------------------

                target_x = (
                    (finger_x - frame_margin)
                    / (width - 2 * frame_margin)
                ) * screen_width


                # -----------------------------------------
                # Map camera Y to screen Y
                # -----------------------------------------

                target_y = (
                    (finger_y - frame_margin)
                    / (height - 2 * frame_margin)
                ) * screen_height


                # -----------------------------------------
                # Keep cursor inside screen
                # -----------------------------------------

                target_x = max(
                    0,
                    min(
                        screen_width - 1,
                        target_x
                    )
                )

                target_y = max(
                    0,
                    min(
                        screen_height - 1,
                        target_y
                    )
                )


                # -----------------------------------------
                # Smooth cursor movement
                # -----------------------------------------

                current_x = (
                    previous_x
                    + (target_x - previous_x)
                    / smoothening
                )

                current_y = (
                    previous_y
                    + (target_y - previous_y)
                    / smoothening
                )


                # -----------------------------------------
                # Move actual Windows cursor
                # -----------------------------------------

                pyautogui.moveTo(
                    current_x,
                    current_y
                )


                # Save current position
                previous_x = current_x
                previous_y = current_y


                # Display mapped screen position
                cv2.putText(
                    frame,
                    f"Mouse: ({int(current_x)}, "
                    f"{int(current_y)})",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


    # =====================================================
    # 8. DISPLAY
    # =====================================================

    cv2.imshow(
        "Smart Virtual Mouse",
        frame
    )


    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# 9. CLEANUP
# =========================================================

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Program closed.")