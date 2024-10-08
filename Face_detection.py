import cv2
import datetime
import os


harcascade = "haar_casscade.xml"

if not os.path.exists('Captured'):
    os.makedirs('Captured')

cap = cv2.VideoCapture(0)

cap.set(3, 2000) #width of the frame
cap.set(4, 1500) #Height of the frame

while True:
    success, img = cap.read()

    if not success:
        break


    facecascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    face = facecascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"Captured/captured_face_{timestamp}.jpg"
        
        # Save the frame with the unique filename in the 'Captured' folder
        cv2.imwrite(filename, img)
        print(f"Face captured and saved as {filename}")


    cv2.imshow("face", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
