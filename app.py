import cv2  as cv
import datetime
import time
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PySide6.QtGui import QPixmap, QIcon, QPalette, QColor

# Title for datas
titles = ["Name", "Blood", "Experience(km)", "Accident"]
# People who allowed to drive the car and datas
people = [["Ali Emre Kaya", "0+", 134, 0]]

# Load a trained face recognizer model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")
# Load OpenCV's pre-trained Haar cascade for face detection
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# THE APPLICATION
app = QApplication(sys.argv)
window = QMainWindow()
window.setGeometry(100, 100, 600, 500)
window.setWindowTitle("Driver Control")
app.setWindowIcon(QIcon("images/icon.ico"))
window.setMinimumSize(600, 500)
window.setMaximumSize(600, 500)

# Set background image
background_image = QPixmap("images/background.png")
background_label = QLabel(window)
background_label.setPixmap(background_image)
background_label.setGeometry(0, 0, 600, 500)

# Create labels for displaying data
img_width = 1672 / 4
img_height = 1116 / 4
img_label_start_x = (600 - img_width) / 2
img_label = QLabel(window)
img_label.setGeometry(img_label_start_x, 50, img_width, img_height)

# Create a QPalette and set the text color (foreground color)
palette = QPalette()
text_color = QColor(0, 0, 0)  
palette.setColor(QPalette.WindowText, text_color)

# Drive's datas
text1 = QLabel(window)
text1.setGeometry(120, 310, 300, 30)
text1.setPalette(palette)
text2 = QLabel(window)
text2.setGeometry(150, 340, 300, 30)
text2.setPalette(palette)
text3 = QLabel(window)
text3.setGeometry(180, 360, 300, 30)
text3.setPalette(palette)
text4 = QLabel(window)
text4.setGeometry(180, 380, 300, 30)
text4.setPalette(palette)
text5 = QLabel(window)
text5.setGeometry(180, 400, 300, 30)
text5.setPalette(palette)


def take_photo():
    cap = cv.VideoCapture(0)

    # Give enough time to camera for opening 
    time.sleep(1)
    ret, frame = cap.read()   

    # Save frame in current_driver folder as jpg file 
    count = 0
    cv.imwrite("current_driver/frame%d.jpg" % count, frame)
    image_name = "frame%d" % count
    
    img = cv.imread(f"current_driver/{image_name}.jpg")
    pure_img = img

    # Get the height and width of the image
    height, width = img.shape[:2]

    # Convert the image to grayscale for better analysis
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

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
            label, not_confidence = face_recognizer.predict(faces_roi)

            if(not_confidence < confidence_ratio):
                confidence_ratio = not_confidence

            # Define font properties for displaying text, general
            font_scale = height / 800.0
            thickness = max(1, int(height / 300))
            not_confidence_size = cv.getTextSize(str(not_confidence), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            start_x_conf = int((x + (w/2)) - not_confidence_size[1] / 2)

            # Driving permission given
            if(confidence_ratio < 22):
                # Define font properties for displaying text when if condition true
                text = people[label][0]
                text_size = cv.getTextSize(str(text), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2)[0]
                start_x = int((x + (w/2)) - text_size[0] / 2)
                # Display the name above and below the detected face
                cv.putText(img, str(text), (start_x, y - int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=2)
                cv.putText(img, str(int(not_confidence)), (start_x_conf, y + h + not_confidence_size[1] + int(height/50)), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=2)

                # Draw a rectangle around the detected face
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                # Output
                print(f"{text} : {not_confidence}")

                # Save image for better confidence (I'M NOT SURE ON THIS PART) that may be Overfitting
                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                output_path = f"dataset/{people[label][0]}/{timestamp}.jpg"
                cv.imwrite(output_path, pure_img)

                # Save frame in past_drives permanently as succesful
                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                output_path = f"past_drives/{timestamp}_YES.jpg"
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
                img_label.setPixmap(pixmap)

                # Drive datas
                text1_ = f"Allowed to drive, {people[label][0]} can start the car."
                text1.setText(text1_)
                text2_ = f"{people[label][0]}'s Informations: "
                text2.setText(text2_)
                text3_ = f"{titles[1]} : {people[label][1]}"
                text3.setText(text3_)
                text4_ = f"{titles[2]} : {people[label][2]}"
                text4.setText(text4_)
                text5_ = f"{titles[3]} : {people[label][3]}"
                text5.setText(text5_)

                print(f"Allowed to drive, {people[label][0]} can start the car.")

                print("----------------------------------------------")
                print(f"{people[label][0]}'s Informations: ")
                for i in range(1, len(people[label])):
                    print(f"{titles[i]} : {people[label][i]}")
                    
            
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

                # Output
                print(f"{text} : {not_confidence}")

                # Save frame in past_drives permanently as unsuccesful
                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                output_path = f"past_drives/{timestamp}_NO.jpg"
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
                img_label.setPixmap(pixmap)

                text1_ = "Unknown People"
                text1.setText(text1_)
                text2_ = ""
                text2.setText(text2_)
                text3_ = ""
                text3.setText(text3_)
                text4_ = ""
                text4.setText(text4_)
                text5_ = ""
                text5.setText(text5_)

                print("Not allowed to drive.")

    
    # Permission to access camera
    cap.release()

    # Close the camera
    cv.destroyAllWindows()
    


# Button for visualize the image
button_run = QPushButton("Run", window)
button_run.setGeometry(250, 30, 100, 30)
button_run.setStyleSheet("background-color: green;")
button_run.clicked.connect(take_photo)

# Show all app
window.show()

# Closed the app
sys.exit(app.exec())