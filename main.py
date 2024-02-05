import cv2
import os
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import csv
import joblib

nimgs = 5

current_time = datetime.now().strftime("%H:%M:%S")
current_month = date.today().strftime("%B")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def totalreg():
    return len(os.listdir('faces'))


def train_model(program_name):
    faces = []
    labels = []
    userlist = [user for user in os.listdir(f'face/{program_name}') if os.path.isdir(os.path.join('face', program_name, user))]

    if not userlist:
        print("No faces found for training.")
        return None

    for user in userlist:
        for imgname in os.listdir(os.path.join(f'face/{program_name}', user)):
            img = cv2.imread(os.path.join(f'face/{program_name}', user, imgname))
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)

    if len(set(labels)) > 1:  # Check if there are at least two unique labels for training
        model_save_dir = 'face'
        os.makedirs(model_save_dir, exist_ok=True)  # Create 'face' directory if it doesn't exist
        knn.fit(faces, labels)
        model_save_path = os.path.join(model_save_dir, 'face_recognition_model.pkl')
        joblib.dump(knn, model_save_path)
        return knn
    else:
        print("Insufficient data for training. At least two individuals required.")
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


def save_attendance(program_name, subject, username, userid, timestamp):
    Attendance_path = "Attendance"
    os.makedirs(Attendance_path, exist_ok=True)
    csv_file_path = os.path.join('Attendance', f"Attendance for {program_name} subject {subject} in {current_month}.csv")
    with open(csv_file_path, "a", newline="\n") as f:
        lnwrite = csv.writer(f)
        lnwrite.writerow([subject, username, userid, timestamp])


def start():
    program_name = input('Enter Your Program name: ')
    model = train_model(program_name)
    subject = input('Enter your Subject for Attendance: ')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()

        # Check if the frame has a valid size
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:

            if len(extract_faces(frame)) > 0:
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1), model)
                if identified_person:
                    add(identified_person, program_name, subject)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                    cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()



def add(new_user, program_name, subject):
    new_user_name, new_user_id = new_user.split('_')

    user_image_folder = os.path.join('face', f'{program_name}', f'{new_user_name}_{new_user_id}')
    if not os.path.isdir(user_image_folder):
        os.makedirs(user_image_folder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = f'{new_user_name}_{new_user_id}_{i}.jpg'
                cv2.imwrite(os.path.join(user_image_folder, name), frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if i == nimgs:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_attendance(program_name, subject, new_user_name, new_user_id, timestamp)


if __name__ == "__main__":
    # Choose the operation to perform: 'add' to add a new user, 'start' to start attendance
    operation = input("Enter operation ('add' or 'start'): ")

    if operation == 'add':
        program_name = input('Enter your Program Name: ')
        new_user_name = input('Enter new username: ')
        new_user_id = input('Enter new user ID: ')

        add(f'{new_user_name}_{new_user_id}_{program_name}', program_name)

    elif operation == 'start':
        start()
    else:
        print("Invalid operation. Please enter 'add' or 'start'.")
