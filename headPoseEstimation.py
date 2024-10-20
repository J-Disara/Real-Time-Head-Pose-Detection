import cv2
import mediapipe as mp
import numpy as np
import time

def adjust_brightness_contrast(image, brightness=0, contrast=30):
    # Apply brightness and contrast correction
    image = cv2.convertScaleAbs(image, alpha=1 + contrast / 127.0, beta=brightness)
    return image

def denoise_image(image):
    # Apply fast Non-Local Means Denoising
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def sharpen_image(image):
    # Define sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opened.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Frame capture failed.")
        break

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    nose_2d = None
    nose_3d = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = img_w  # Adjusted focal length
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            if success:
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxr, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                z_angle = angles[2] * 360

                # Determine head direction
                if y_angle < -10:
                    text = "Looking Left"
                elif y_angle > 10:
                    text = "Looking Right"
                elif x_angle < -10:
                    text = "Looking Down"
                elif x_angle > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the nose direction
                if nose_3d is not None:
                    nose_3d_projection, jacobin = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add text on the image for the current frame
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break
        
cap.release()
cv2.destroyAllWindows()