import os
import cv2 as cv
import datetime
import time
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PySide6.QtGui import QPixmap, QIcon, QPalette, QColor, QFont
from PySide6.QtCore import QTimer
from nbconvert import PythonExporter
from nbformat import read
from deepface import DeepFace

class DriverControlApp(QMainWindow):
    titles = ["Name", "Blood", "Experience(km)", "Accident"]
    people = [["Ali Emre Kaya", "0+", 134, 0],
              ["Burhan Altintop", "A+", 0, 3]]

    def __init__(self):
        super().__init__()

        self.label = ""
        self.risk_value = 1

        # Load a trained face recognizer model
        self.face_recognizer = cv.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read("data/face_trained.yml")
        # Load OpenCV's pre-trained Haar cascade for face detection
        self.haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Application UI setup
        self.setup_ui()

        self.is_successful = False
        self.timer = QTimer()
        if not self.is_successful:
            self.timer.timeout.connect(self.timer_callback)
            self.timer.start(1000)

    def setup_ui(self):
        self.setGeometry(100, 100, 600, 500)
        self.setWindowTitle("Driver Control")
        self.setWindowIcon(QIcon("images/icon.ico"))
        self.setMinimumSize(600, 500)
        self.setMaximumSize(600, 500)

        # Set background image
        background_image = QPixmap("images/background.png")
        background_label = QLabel(self)
        background_label.setPixmap(background_image)
        background_label.setGeometry(0, 0, 600, 500)

        # Create labels for displaying data
        img_width = 1672 / 4
        img_height = 1116 / 4
        img_label_start_x = (600 - img_width) / 2
        self.img_label = QLabel(self)
        self.img_label.setGeometry(img_label_start_x, 50, img_width, img_height)

        # Create a QPalette and set the text color (foreground color)
        self.palette = QPalette()
        text_color = QColor(0, 0, 0)
        self.palette.setColor(QPalette.WindowText, text_color)

        self.palette_red = QPalette()
        text_color_red = QColor(255, 0, 0)
        self.palette_red.setColor(QPalette.WindowText, text_color_red)

        font = QFont()
        font.setPointSize(24) 

        # Drive's datas
        self.text1 = QLabel(self)
        self.text1.setGeometry(120, 310, 300, 30)
        self.text1.setPalette(self.palette)
        self.text2 = QLabel(self)
        self.text2.setGeometry(150, 340, 300, 30)
        self.text2.setPalette(self.palette)
        self.text3 = QLabel(self)
        self.text3.setGeometry(180, 360, 300, 30)
        self.text3.setPalette(self.palette)
        self.text4 = QLabel(self)
        self.text4.setGeometry(180, 380, 300, 30)
        self.text4.setPalette(self.palette)
        self.text5 = QLabel(self)
        self.text5.setGeometry(180, 400, 300, 30)
        self.text5.setPalette(self.palette)
        self.text6 = QLabel(self)
        self.text6.setGeometry(180, 440, 300, 30)
        self.text6.setPalette(self.palette)
        self.text6.setFont(font)

    def take_photo(self):
        cap = cv.VideoCapture(0)

        # Give enough time to camera for opening 
        time.sleep(1)
        ret, frame = cap.read() 

        # Permission to access camera
        cap.release()

        # Save frame in current_driver folder as jpg file 
        count = 0
        cv.imwrite("current_driver/frame%d.jpg" % count, frame)
        image_name = "frame%d" % count

        img = cv.imread(f"current_driver/{image_name}.jpg")
        pure_img = img

        return img, pure_img

    def run(self):
        img, pure_img = self.take_photo()

        # Get the height and width of the image
        height, width = img.shape[:2]

        # Convert the image to grayscale for better analysis
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detect faces using the Haar cascade
        faces_rect = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if(len(faces_rect) == 0):
            print("FAIL, no face detected.")

        else:
            # Selecting the biggest face
            # (assuming that will be the inividual)
            biggest_rect = -1
            xB = 0
            yB = 0
            wB = 0
            hB = 0
            for(x, y, w, h) in faces_rect:
                new_rect = abs(w)
                if(biggest_rect < new_rect):
                    biggest_rect = new_rect
                    xB = x
                    yB = y
                    wB = w
                    hB = h

            # Indexes of biggest face
            faces_rect = [[xB, yB, wB, hB]]

            # Controlling the confidence
            confidence_ratio = 200

            # Rectangle and Text
            for (x, y, w, h) in faces_rect:
                # Extract the region of interest (ROI) for face recognition
                faces_roi = gray[y:y+h, x:x+w]

                # Perform face recognition to determine the person and confidence level
                self.label, not_confidence = self.face_recognizer.predict(faces_roi)

                if(not_confidence < confidence_ratio):
                    confidence_ratio = not_confidence

                # Define font properties for displaying text, general
                font_scale = height / 800.0
                thickness = max(1, int(height / 300))
                not_confidence_size = cv.getTextSize(str(not_confidence), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                start_x_conf = int((x + (w/2)) - not_confidence_size[1] / 2)

                # Driving permission given
                if(confidence_ratio < 32):
                    self.is_successful = True
                    # Define font properties for displaying text when if condition true
                    text = self.people[self.label][0]
                    text_size = cv.getTextSize(str(text), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)[0]
                    start_x = int((x + (w/2)) - text_size[0] / 2)
                    # Display the name above and below the detected face
                    cv.putText(img, str(text), (start_x, y - int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=2)
                    cv.putText(img, str(int(not_confidence)), (start_x_conf, y + h + not_confidence_size[1] + int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=2)

                    # Draw a rectangle around the detected face
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                    # Output
                    print(f"{text} : {not_confidence}")

                    # Save image for better confidence
                    directory_path = f'dataset/{self.people[self.label][0]}'
                    file_count = 0
                    deleted_file = ""
                    deleted_path = f"dataset/{self.people[self.label][0]}/{deleted_file}"
                    for root, dirs, files in os.walk(directory_path):
                        file_count += len(files)
                        deleted_file = min(files)
                        deleted_path = f"dataset/{self.people[self.label][0]}/{deleted_file}"
                        if(deleted_file == ".DS_Store"):
                            os.remove(deleted_path)
                            files.remove(deleted_file)
                            deleted_file = min(files)
                            deleted_path = f"dataset/{self.people[self.label][0]}/{deleted_file}"

                    if(file_count < 50):
                        current_time = datetime.datetime.now()
                        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                        output_path = f"dataset/{self.people[self.label][0]}/{timestamp}.jpg"
                        cv.imwrite(output_path, pure_img)
                    else:
                        deleted_path = f"dataset/{self.people[self.label][0]}/{deleted_file}"
                        os.remove(deleted_path)
                        current_time = datetime.datetime.now()
                        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                        output_path = f"dataset/{self.people[self.label][0]}/{timestamp}.jpg"
                        cv.imwrite(output_path, pure_img)


                    # Save frame in past_drives permanently as succesful
                    current_time = datetime.datetime.now()
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    output_path = f"past_drives/successful/{timestamp}_{self.people[self.label][0]}.jpg"
                    cv.imwrite(output_path, img)


                    # Arrange image for App
                    desired_width = int(1672 / 4)
                    ratio = height / width
                    desired_height = int(desired_width * ratio)
                    new_size = (desired_width, desired_height)
                    resized_image = cv.resize(img, new_size)
                    temp_output_path = "current_driver/temp_image.jpg"
                    cv.imwrite(temp_output_path, resized_image)

                    # Show image on App
                    pixmap = QPixmap(temp_output_path)
                    self.img_label.setPixmap(pixmap)

                    # Drive datas
                    text1_ = f"Allowed to drive, {self.people[self.label][0]} can start the car."
                    self.text1.setText(text1_)
                    text2_ = f"{self.people[self.label][0]}'s Informations: "
                    self.text2.setText(text2_)
                    text3_ = f"{self.titles[1]} : {self.people[self.label][1]}"
                    self.text3.setText(text3_)
                    text4_ = f"{self.titles[2]} : {self.people[self.label][2]}"
                    self.text4.setText(text4_)
                    text5_ = f"{self.titles[3]} : {self.people[self.label][3]}"
                    self.text5.setText(text5_)

                    print(f"Allowed to drive, {self.people[self.label][0]} can start the car.")

                    print("----------------------------------------------")
                    print(f"{self.people[self.label][0]}'s Informations: ")
                    for i in range(1, len(self.people[self.label])):
                        print(f"{self.titles[i]} : {self.people[self.label][i]}")

                    # Execute new recognizer model with new data
                    ipynb_path = "train.ipynb"
                    with open(ipynb_path, 'r', encoding='utf-8') as nb_file:
                        notebook = read(nb_file, as_version=4)

                    exporter = PythonExporter()
                    source, _ = exporter.from_notebook_node(notebook)
                    print("New model trained")
                        
                else:
                    # Define font properties for displaying text when if condition false
                    text = "unknown"
                    text_size = cv.getTextSize(str(text), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)[0]
                    start_x = int((x + (w/2)) - text_size[0] / 2)
                    # Display the name above and below the detected face
                    cv.putText(img, str(text), (start_x, y - int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness=2)
                    cv.putText(img, str(int(not_confidence)), (start_x_conf, y + h + not_confidence_size[1] + int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness=2)

                    # Draw a rectangle around the detected face
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

                    # Output
                    print(f"{text} : {not_confidence}")

                    # Save frame in past_drives permanently as unsuccesful
                    current_time = datetime.datetime.now()
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    output_path = f"past_drives/unsuccessful/{timestamp}.jpg"
                    cv.imwrite(output_path, img)

                    # Arrange image for App
                    desired_width = int(1672 / 4)
                    ratio = height / width
                    desired_height = int(desired_width * ratio)
                    new_size = (desired_width, desired_height)
                    resized_image = cv.resize(img, new_size)
                    temp_output_path = "current_driver/temp_image.jpg"
                    cv.imwrite(temp_output_path, resized_image)

                    # Show image on App
                    pixmap = QPixmap(temp_output_path)
                    self.img_label.setPixmap(pixmap)

                    self.text1.setText("Unknown People")
                    self.text2.setText("")
                    self.text3.setText("")
                    self.text4.setText("")
                    self.text5.setText("")

                    print("Not allowed to drive.")      
    
    # define the dominant emotion of the driver
    def emotion_recognizer(self, frame_path):
        faces = DeepFace.extract_faces(frame_path)
        if not faces:
            print("No faces detected in the image.")
            return "None"
        else:
            numpy_array = faces[0]["face"] * 255 
            jpg_filename = "current_driver/face0.jpg"
            cv.imwrite(jpg_filename, numpy_array, [cv.IMWRITE_JPEG_QUALITY, 95])
            
            predictions = DeepFace.analyze(jpg_filename, enforce_detection=False)
            if not predictions:
                print("No emotion predictions available.")
                return "None"
            else:
                return predictions[0]["dominant_emotion"]

    def emotion_run(self):
        time.sleep(15)
        emotion_coef = {
                "angry":20,
                "fear":15,
                "surprise":15,
                "disgust":12,
                "sad":12,
                "happy":1,
                "neutral":1,
                }
        img, pure_img = self.take_photo()
        height, width = img.shape[:2]

        emotion = self.emotion_recognizer(img)
        if emotion != "None":
            # Arrange image for App
            desired_width = int(1672 / 4)
            ratio = height / width
            desired_height = int(desired_width * ratio)
            new_size = (desired_width, desired_height)
            resized_image = cv.resize(img, new_size)
            temp_output_path = "current_driver/temp_emotion_image.jpg"
            cv.imwrite(temp_output_path, resized_image)

            # Show image on App
            pixmap = QPixmap(temp_output_path)
            self.img_label.setPixmap(pixmap)

            self.risk_value = emotion_coef[emotion] * (self.people[self.label][3]+1)
            if(self.risk_value > 15):
                self.text6.setText("You need a break !")
                self.text6.setPalette(self.palette_red)
            else:
                self.text6.setText(f"Driving, risk:{self.risk_value}")
                self.text6.setPalette(self.palette)
    
    def timer_callback(self):
        if not self.is_successful:
            self.run()
        else:
            self.emotion_run()


def main():
    app = QApplication(sys.argv)
    window = DriverControlApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
