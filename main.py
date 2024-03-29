import face_recognition
import cv2
import numpy as np
import datetime
import csv  # Import the csv module for data recording

# Load known faces (modify filenames and names as needed))
known_face_images = ["faceimages/sujitresumephoto.jpg", "faceimages/download.jpg"]
known_face_names = ["Black Shadow", "new"]

# Initialize empty list for face encodings
known_face_encodings = []

# Load and encode known faces
for image_path, name in zip(known_face_images, known_face_names):
    face_image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(face_image)[0]  # Extract first encoding
    known_face_encodings.append(face_encoding)

# Initialize video capture and csv writer
video_capture = cv2.VideoCapture(1)
csv_filename = "attendance_log.csv"  # Define CSV filename

# Create a CSV file with headers (modify if needed)
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Date", "Time", "Name"])

while True:
    _, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # Compare face encoding to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(f"Recognized: {name}")

            # Add text to frame and write to CSV if present
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (25, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            # Only write to CSV if the name is recognized
            if name in known_face_names:
                now = datetime.datetime.now()  # Get current time
                current_time = now.strftime("%H:%M:%S")
                with open(csv_filename, 'a', newline='') as csvfile:  # Append to CSV
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([now.date(), current_time, name])

    # Display frame and handle exit
    cv2.imshow("attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
video_capture.release()
cv2.destroyAllWindows()

print(f"Attendance recorded in {csv_filename}")
