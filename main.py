import cv2
import os
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import csv
import joblib

nimgs = 5

current_date = datetime.now().strftime("%d-%b-%Y")
current_time = datetime.now().strftime("%H:%M:%S")
current_month = date.today().strftime("%B")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def totalreg():
    return len(os.listdir('faces'))

def train_model(program_name):
    faces = []
    labels = []

    try:
        userlist = [user for user in os.listdir(f'face/{program_name}') if
                    os.path.isdir(os.path.join(f'face/{program_name}', user))]

        if not userlist:
            print("No faces found for training.")
            return None

        for user in userlist:
            user_directory = os.path.join(f'face/{program_name}', user)
            for imgname in os.listdir(user_directory):
                img = cv2.imread(os.path.join(user_directory, imgname))
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)

        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)

        if len(set(labels)) > 1:
            model_save_dir = 'face'
            os.makedirs(os.path.join(model_save_dir, program_name), exist_ok=True)
            knn.fit(faces, labels)
            model_save_path = os.path.join(model_save_dir, program_name, 'face_recognition_model.pkl')
            joblib.dump(knn, model_save_path)
            return knn
        else:
            print("Insufficient data for training. At least two individuals required.")
            return None

    except Exception as e:
        print(f"Error during training: {e}")
        return None
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []
    
def identify_face(facearray, model):
    if model is not None:
        return model.predict(facearray.reshape(1, -1))[0]
    else:
        return None

def start():
    program_name = input('Enter Your Program name: ').upper()
    csv_file_path = None

    try:
        model = train_model(program_name)
    except FileNotFoundError as e:
        print(f'Error is:{e}')
        return

    subject = input('Enter your Subject for Attendance: ').capitalize()

    if not os.path.isfile(get_attendance_file_path(program_name, subject)):
        create_attendance_csv(program_name, subject)

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()

        # Check if the frame has a valid size
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:

            if len(extract_faces(frame)) > 0:
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1), model)
                if identified_person:
                    csv_file_path = save_attendance(program_name, subject, identified_person, current_time,
                                                    current_date)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                    cv2.putText(frame, f'{identified_person}', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return csv_file_path


def add(new_user, program_name):
    new_user_name, new_user_id, program_name = new_user.split('_')

    user_image_folder = os.path.join('face', f'{program_name}', f'{new_user_name}_{new_user_id}')
    if not os.path.isdir(user_image_folder):
        os.makedirs(user_image_folder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = f'{new_user_name}_{new_user_id}_{i}.jpg'
                cv2.imwrite(os.path.join(user_image_folder, name), frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if i == nimgs:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def save_attendance(program_name, subject, identified_person, current_time, current_date):
    csv_file_path = get_attendance_file_path(program_name, subject)

    if not os.path.isfile(csv_file_path):
        create_attendance_csv(program_name, subject)

    attendance_data = [identified_person, current_date]

    # Check if the student's attendance has already been recorded for the current date and subject
    if not is_attendance_recorded(csv_file_path, identified_person, current_date):
        with open(csv_file_path, "a", newline="\n") as f:
            lnwrite = csv.writer(f)
            lnwrite.writerow(attendance_data)

    return csv_file_path


def is_attendance_recorded(csv_file_path, student_id, current_date):
    with open(csv_file_path, "r", newline="\n") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            if row[0] == student_id and row[2] == current_date:
                return True
    return False


def get_attendance_file_path(program_name, subject):
    return os.path.join('Attendance', f"Attendance for {program_name} subject {subject} in {current_month}.csv")


def create_attendance_csv(program_name, subject):
    csv_file_path = get_attendance_file_path(program_name, subject)
    header = ["Name", subject, "Date"]
    with open(csv_file_path, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return csv_file_path

def get_attendance(program_name, subject):
    csv_file_path = get_attendance_file_path(program_name, subject)

    try:
        with open(csv_file_path, "r") as f:
            reader = csv.reader(f)
            # Check if the first row contains headers
            header = next(reader)
            if "Name" in header and "Time" in header and "Date" in header:
                # Check if there is at least one data row
                for row in reader:
                    if row:
                        return csv_file_path
        return None
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    # Choose the operation to perform: 'add' to add a new user, 'start' to start attendance
    while True:
        operation = input("Enter operation ('add', 'start', 'get' or 'exit'): ")

        if operation == 'add':
            program_name = input('Enter your Program Name: ').upper()
            new_user_name = input('Enter new username: ').capitalize()
            new_user_id = input('Enter new user ID: ')

            add(f'{new_user_name}_{new_user_id}_{program_name}', program_name)

        elif operation == 'start':
            print(f"Atleast Two student in {program_name} then start:")
            csv_file_path = start()
            print(f"Attendance data saved in: {csv_file_path}")

        elif operation == 'get':
            program_name = input('Enter your Program Name: ').upper()
            subject = input('Enter subject for attendance: ').capitalize()

            attendance_file_path = get_attendance(program_name, subject)
            if attendance_file_path:
                # Read the file or perform any other operations with the file path
                print(f"Attendance file found at: {attendance_file_path}")
            else:
                print("Attendance file not found.")

        elif operation == 'exit':
            break

        else:
            print("Invalid operation. Please enter 'add', 'start', 'get' or 'exit'.")
