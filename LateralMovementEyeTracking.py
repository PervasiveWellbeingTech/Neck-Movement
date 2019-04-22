import cv2
import sys
import numpy as np

def detect():
    margin = 100

    faceCascade = cv2.CascadeClassifier("databases/haarcascade_frontalface_alt2.xml")
    #eyeCascade = cv2.CascadeClassifier("databases/haarcascade_eye_tree_eyeglasses.xml")
    eyeCascade = cv2.CascadeClassifier("databases/haarcascade_eye.xml")

    camera = cv2.VideoCapture("c112ft.mp4")
    file = open("C112front.txt", 'w')
    prevAngle = None
    prevTime = None
    count = 0
    validcount = 0
    totalcount = 0

    startUnix = 1533137408


    while (True):
        face = False
        eye = False

        ret, frame = camera.read()
        angle = 0
        velocity = 0

        if not ret:
            break

        if frame is not None:
            frame = frame[330:580, 760:1290]
            #frame = frame[0:1440, 930:1030]
            #frame = cv2.addWeighted(frame, 1.5, np.zeros(frame.shape, frame.dtype), 0, 60)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #faces = faceCascade.detectMultiScale(gray, 1.3, 8, minSize = (50, 50))
            #for (x, y, w, h) in faces:
                #face = True
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #roi_gray = gray[y:y + h + margin, x:x + w + margin]
                #roi_color = frame[y:y + h + margin, x:x + w + margin]

            eyes = eyeCascade.detectMultiScale(gray, minNeighbors = 18, minSize = (40, 40))

            #print(np.size(eyes) == 0)
            #print(np.shape(eyes))

            leftX = 0
            leftY = 0
            rightX = 0
            rightY = 0
            first = True
            second = False

            firstX = 0
            firstY = 0
            secondX = 0
            secondY = 0

            for (ex, ey, ew, eh) in eyes:
                if second :
                    break
                eyeX = ex + ew/2
                eyeY = ey + eh/2

                if first:
                    firstX = eyeX
                    firstY = eyeY
                    first = False

                else:
                    secondX = eyeX
                    secondY = eyeY
                    second = True
                    eye =True
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
                #print(ew, eh)

                if eye:
                    if (firstX > secondX):
                        leftX = firstX
                        leftY = firstY
                        rightX = secondX
                        rightY = secondY
                    else:
                        rightX = firstX
                        rightY = firstY
                        leftX = secondX
                        leftY = secondY
                    angle = np.arctan((leftY - rightY) / (leftX - rightX))

            cv2.imshow("camera", frame)

            currTime = camera.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = startUnix + currTime / 1000

            file.write((str)(timestamp))
            file.write(' ')
            file.write((str)(angle))
            file.write(' ')
            if eye:
                if prevTime is not None:
                    velocity = (angle - prevAngle) / (currTime - prevTime)
                prevAngle = angle
                prevTime = currTime
            else:
                prevAngle = None
                prevTime = None
            file.write((str)(velocity))
            file.write('\n')


        totalcount += 1
        if eye:
            validcount +=1

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    print("valid: " + (str)(validcount))
    print((str)(validcount / totalcount * 100) + "%")
    camera.release()
    file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()

