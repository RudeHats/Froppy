import cv2
import dlib
import numpy as np

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# To hold unique face encodings
unique_faces = set()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        shape = predictor(gray, face)
        face_embedding = np.array(recognition_model.compute_face_descriptor(frame, shape))

        # Check if this face is already counted
        if tuple(face_embedding) not in unique_faces:
            unique_faces.add(tuple(face_embedding))
    
    # Display the video feed
    cv2.imshow("Webcam", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the total number of unique faces
print("Number of unique faces:", len(unique_faces))

# Release resources
cap.release()
cv2.destroyAllWindows()
