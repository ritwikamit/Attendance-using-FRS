import face_recognition
import cv2
import numpy as np
import os
import csv
import threading
import ast

# Global synchronization and shared variables
exit_event = threading.Event()
capture_lock = threading.Lock()
last_frame = None  # The latest frame from the camera

# Configuration paths
save_directory = r"c:/Users/HP/OneDrive/Desktop/project/attendance/photos/"
registered_faces_file = os.path.join(save_directory, "registered_faces.csv")

# Global known faces lists
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load registered face data from CSV and update known_face_encodings and known_face_names."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    if os.path.exists(registered_faces_file):
        try:
            with open(registered_faces_file, newline="") as csvfile:
                csvreader = csv.reader(csvfile)
                header = next(csvreader, None)  # Skip header if present
                for row in csvreader:
                    if len(row) < 2:
                        continue
                    name = row[0]
                    encoding_str = row[1]
                    # Convert the string representation back to a list, then to a NumPy array.
                    encoding = np.array(ast.literal_eval(encoding_str))
                    known_face_names.append(name)
                    known_face_encodings.append(encoding)
            print(f"Loaded {len(known_face_names)} registered faces.")
        except Exception as e:
            print("Error loading known faces:", e)
    else:
        print("No registered faces found.")

def camera_thread():
    """Continuously capture frames from the webcam and update the global last_frame."""
    global last_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        exit_event.set()
        return
    while not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        with capture_lock:
            last_frame = frame.copy()
    cap.release()

def save_new_face(name, encoding):
    """Save a new face's name and encoding to the CSV file."""
    try:
        file_exists = os.path.isfile(registered_faces_file)
        with open(registered_faces_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Encoding"])
            writer.writerow([name, encoding.tolist()])
        return True
    except Exception as e:
        print(f"Error saving face data: {e}")
        return False

def register_new_face():
    """
    Allow registration of a new face by:
      - Pressing 'C' to capture the face.
      - Then typing the name (handles backspace, etc.).
      - Press ENTER to save the registration.
      - Press Q at any moment to cancel.
    """
    print("Entering face registration mode.")
    name = ""
    capture_mode = True
    face_encoding = None

    # Grab the current frame for registration.
    with capture_lock:
        if last_frame is None:
            print("No frame available for registration.")
            return False
        reg_frame = last_frame.copy()

    while not exit_event.is_set():
        # Update registration frame from the latest available frame.
        with capture_lock:
            if last_frame is not None:
                reg_frame = last_frame.copy()
        display_frame = reg_frame.copy()

        if capture_mode:
            # In capture mode, instruct the user to press 'C' and show bounding box if a face is detected.
            cv2.putText(display_frame, "Press 'C' to Capture Face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Scale down the frame for detection for performance.
            small_frame = cv2.resize(display_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small)
            if face_locations:
                # Draw the bounding box (scaled back up) for the first detected face.
                top, right, bottom, left = [coord * 4 for coord in face_locations[0]]
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            # In name entry mode, show the current name being typed.
            cv2.putText(display_frame, f"Enter Name: {name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press ENTER to Save", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Face Registration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if capture_mode:
            if key == ord('c'):
                if not face_locations:
                    print("No face detected. Try again.")
                    continue
                # Obtain the encoding for the first detected face.
                face_encs = face_recognition.face_encodings(rgb_small, face_locations)
                if face_encs:
                    face_encoding = face_encs[0]
                    capture_mode = False
                    print("Face captured! Now, type your name.")
                else:
                    print("Failed to encode face. Try again.")
            elif key == ord('q'):
                cv2.destroyWindow("Face Registration")
                return False
        else:
            if key == 13:  # ENTER key
                if name and (face_encoding is not None):
                    try:
                        image_path = os.path.join(save_directory, f"{name}.jpg")
                        cv2.imwrite(image_path, reg_frame)
                        if save_new_face(name, face_encoding):
                            print(f"Face registered successfully as '{name}'.")
                            cv2.destroyWindow("Face Registration")
                            return True
                        else:
                            print("Error: Failed to save registration data.")
                            return False
                    except Exception as e:
                        print("Error during registration:", e)
                        return False
            elif key == 8:  # Backspace key
                name = name[:-1]
            elif 32 <= key <= 126:  # Append printable characters
                name += chr(key)
            elif key == ord('q'):
                cv2.destroyWindow("Face Registration")
                return False

    cv2.destroyWindow("Face Registration")
    return False

def main():
    """Main loop: display live feed with recognition and handle key events."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    load_known_faces()  # Load previously registered faces

    # Start camera capture thread.
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    print("Face Registration System")
    print("Press 'R' to register a new face.")
    print("Press 'Q' to quit.")

    while not exit_event.is_set():
        with capture_lock:
            if last_frame is None:
                continue
            main_frame = last_frame.copy()

        # --- Face Recognition Process ---
        if known_face_encodings:
            small_frame = cv2.resize(main_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings_in_frame):
                top, right, bottom, left = [coord * 4 for coord in face_location]
                # Compare with known faces, you can adjust tolerance if needed.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                if any(matches):
                    first_match_index = matches.index(True)
                    name_found = known_face_names[first_match_index]
                    label = f"{name_found} has registered"
                else:
                    label = "Unregistered Face"
                cv2.rectangle(main_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(main_frame, label, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

        # --- End Recognition Process ---

        cv2.imshow("Face Registration - Main Window", main_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            # Close the main window so registration can run on its own.
            cv2.destroyWindow("Face Registration - Main Window")
            if register_new_face():
                print("Registration successful!")
                load_known_faces()  # Reload the registered faces list after a new registration.
            else:
                print("Registration cancelled or failed.")
        elif key == ord('q'):
            exit_event.set()

    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main()
