import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Height

# Load Haar Cascade for faster face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# List of known face encodings and names
known_face_encodings = []
known_face_names = []

# Function to load and encode an image
def load_and_encode_image(image_path, name):
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:  # Ensure at least one face is detected
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)
        else:
            print(f"Warning: No face detected in {image_path}. Skipping this image.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Load and encode face images with their absolute paths and names
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/jobs.jpg", "Jobs")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/ratan_tata.jpg", "Ratan Tata")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/sadmona.jpg", "Sadmona")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/tesla.jpg", "Tesla")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/mypicred.jpg", "Amit Chauhan")

# Prepare for attendance tracking
students = known_face_names.copy()
current_date = datetime.now().strftime("%Y-%m-%d")

# Create and open the CSV file to log attendance
attendance_file = open(f"{current_date}_attendance.csv", "w", newline="")
lnwriter = csv.writer(attendance_file)
lnwriter.writerow(["Name", "Time"])  # Add header to the CSV file

# Main loop for face recognition and attendance tracking
frame_counter = 0
try:
    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame from the webcam.")
            break

        # Process only every 5th frame to reduce lag
        if frame_counter % 5 == 0:
            # Convert the frame to grayscale for faster face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            face_locations = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Convert Haar Cascade results to face_recognition format
            face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]

            # Compute encodings for detected faces
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    # Mark attendance for the recognized person
                    if name in students:
                        students.remove(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"{name} marked present at {current_time}")
                        lnwriter.writerow([name, current_time])

                face_names.append(name)

            # Draw rectangles around faces and label them
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Choose rectangle color (green for known, red for unknown)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display the frame
        cv2.imshow("Attendance", frame)

        # Increment frame counter
        frame_counter += 1

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting gracefully...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
    attendance_file.close()
    print("Attendance tracking program ended gracefully.")