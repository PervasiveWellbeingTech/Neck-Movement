import cv2
import sys
import numpy as np

def detect():

    noseCascade = cv2.CascadeClassifier("databases/haarcascade_mcs_nose.xml")
    mouthCascade = cv2.CascadeClassifier("databases/haarcascade_mcs_mouth.xml")

    camera = cv2.VideoCapture("c107front.mp4")
    file = open("nmfile.txt", 'w')
    prevAngle = None
    prevTime = None
    validcount = 0
    totalcount = 0


    while (True):
        nose = False
        mouth = False
        valid = False

        ret, frame = camera.read()
        angle = 0

        if not ret:
            break

        if frame is not None:
            #frame = frame[200:300, 970:1170]
            frame = frame[600:870, 1050:1500]
            frame = cv2.addWeighted(frame, 1, np.zeros(frame.shape, frame.dtype), 0, 5)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            noses = noseCascade.detectMultiScale(gray, minNeighbors = 8, minSize = (40, 40))
            mouths = mouthCascade.detectMultiScale(gray, minNeighbors = 35, minSize = (60, 60))

            noseX = 0
            noseY = 0
            mouthX = 0
            mouthY = 0

            for (nx, ny, nw, nh) in noses:
                nose = True

                noseX = nx + nw/2
                noseY = ny + nh/2

                cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
                break

            for (mx, my, mw, mh) in mouths:
                if (my > noseY):
                    mouth = True

                    mouthX = mx + mw/2
                    mouthY = my + mh/2

                    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
                    break

            cv2.imshow("camera", frame)

            if nose and mouth and noseY != mouthY:
                valid = True
                angle = np.arctan((noseX - mouthX) / (noseY - mouthY))

            currTime = camera.get(cv2.CAP_PROP_POS_MSEC)
            file.write((str)(currTime) + ' ' + (str)(angle))
            if prevAngle is not None and prevTime is not None:
                if nose and mouth:
                    velocity = (angle - prevAngle) / (currTime - prevTime)
                    file.write(' ' + (str)(velocity))
            prevAngle = angle
            prevTime = currTime
            file.write('\n')

        totalcount += 1
        if valid:
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

