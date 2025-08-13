import face_recognition # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import csv
from datetime import datetime

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

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
#load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/sadmona.jpg", "Sadmona")
#load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/tesla.jpg", "Tesla")
#load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/cap.jpg", "Atul Prakash")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/denku.jpg", "Anshuman Panda")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/komal.jpg", "Aayush Yadav")
#load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/lilpu.jpg", "Alok Kumar")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/jitesh.jpg", "Nitesh Kumar")
#load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/indri.jpg", "Praful kumar")
#load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/sai.jpg", "Sai bathula")
load_and_encode_image("c:/Users/HP/OneDrive/Desktop/project/attendance/photos/mypicred.jpg", "Amit Chauhan")

# Prepare for attendance tracking
students = known_face_names.copy()
current_date = datetime.now().strftime("%Y-%m-%d")

# Create and open the CSV file to log attendance
with open(f"{current_date}_attendance.csv", "w", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])  # Add header to the CSV file

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to access the webcam.")
            break

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Ensure the program doesn't crash if no face is detected
        if not face_locations:
            # Show the video feed even when no faces are detected
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                break
            continue

        # Compute encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []  # Store names of detected faces

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
            # Scale back up face coordinates
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Choose rectangle color (green for known, red for unknown)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Attendance", frame)

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting gracefully...")
            break

# Release resources and close all windows
video_capture.release()
cv2.destroyAllWindows()
print("Attendance tracking program ended gracefully.")