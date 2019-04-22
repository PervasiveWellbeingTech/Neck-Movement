import cv2
import sys
import numpy as np

def detect():
    prevPos = None
    prevTime = None
    #earCascade = cv2.CascadeClassifier("databases/haarcascade_mcs_rightear.xml")
    earCascade = cv2.CascadeClassifier("databases/haarcascade_mcs_leftear.xml")
    camera = cv2.VideoCapture("c112stc3.mp4")
    file = open("C112stc3.txt", 'w')
    first = True

    validcount = 0
    totalcount = 0

    startUnix = 1533137408

    while (True):

        ret, frame = camera.read()

        if not ret:
            break

        velocity = 0
        change = 0
        currPos = 0

        if frame is not None:
            #quad
            #frame = frame[470:520, 850:1000]

            #c5
            #frame = frame[150:600, 850:1650]

            #c3
            frame = frame[360:530, 580:910]

            frame = cv2.addWeighted(frame, 1.5, np.zeros(frame.shape, frame.dtype), 0, 10)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #for (x, y, w, h) in faces:
            #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #    roi_gray = gray[y:y + h + margin, x:x + w + margin]
            #    roi_color = frame[y:y + h + margin, x:x + w + margin]

            ears = earCascade.detectMultiScale(gray, minNeighbors = 3, minSize = (35, 60))

            ear = False
            for (nx, ny, nw, nh) in ears:
                if ear:
                    break
                cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
                currPos = (nx + nw / 2)
                ear = True


            cv2.imshow("camera", frame)

            currTime = camera.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = startUnix + currTime / 1000
            file.write((str)(timestamp))
            file.write(' ')
            file.write((str)(currPos))
            file.write(' ')

            if ear:
                if prevPos is not None:
                    change = currPos - prevPos
                    velocity = change/ (currTime - prevTime)
                prevTime = currTime
                prevPos = currPos
                validcount += 1
            else:
                prevPos = None
                prevTime = None

            file.write((str)(change))
            file.write(' ')
            file.write((str)(velocity))

            file.write('\n')

        totalcount += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("valid: "+(str)(validcount))
    print("Percent:" + (str)(validcount/totalcount*100) + "%")

    camera.release()
    file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()

