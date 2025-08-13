import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# List of known face encodings and names
known_face_encodings = []
known_face_names = []

# File to store registered faces
registered_faces_file = r"c:/Users/HP/OneDrive/Desktop/project/attendance/photos/registered_faces.csv"

# Function to load registered faces from a file
def load_registered_faces():
    if os.path.exists(registered_faces_file):
        with open(registered_faces_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                name, encoding = row[0], np.array(eval(row[1]))
                known_face_encodings.append(encoding)
                known_face_names.append(name)
        print("Loaded registered faces from file.")
    else:
        print("No registered faces file found. Starting with an empty list.")

# Function to save a new face to the registered faces file
def save_new_face(name, encoding):
    with open(registered_faces_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, encoding.tolist()])
    print(f"Saved new face: {name}")

# Function to register a new face
def register_new_face():
    print("Starting face registration...")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame from the webcam.")
            return

        # Display the frame for face capture
        cv2.imshow("Register New Face (Press 'C' to Capture)", frame)

        # Wait for user input
        key = cv2.waitKey(1)
        if key == ord('c'):  # Capture the frame
            break
        elif key == ord('q'):  # Quit registration
            print("Face registration cancelled.")
            return

    # Detect and encode the face
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)

    if not face_locations:
        print("No face detected in the captured frame. Please try again.")
        return

    # Encode the face
    face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]

    # Ask for the name of the new face
    name = input("Enter the name of the new face: ").strip()
    if not name:
        print("No name provided. Registration cancelled.")
        return

    # Save the new face image
    save_directory = r"c:/Users/HP/OneDrive/Desktop/project/attendance/photos/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)  # Create the directory if it doesn't exist

    image_filename = os.path.join(save_directory, f"{name}.jpg")
    cv2.imwrite(image_filename, frame)
    print(f"New face image saved to: {image_filename}")

    # Save the new face encoding
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    save_new_face(name, face_encoding)
    print(f"New face registered: {name}")

# Load registered faces
load_registered_faces()

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

        if not face_locations:
            cv2.imshow("Attendance (Press 'R' to Register New Face)", frame)
            key = cv2.waitKey(1)
            if key == ord('r'):  # Register a new face
                register_new_face()
            elif key == ord('q'):  # Exit if 'q' is pressed
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

        cv2.imshow("Attendance (Press 'R' to Register New Face)", frame)

        # Exit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting gracefully...")
            break

# Release resources and close all windows
video_capture.release()
cv2.destroyAllWindows()
print("Attendance tracking program ended gracefully.")