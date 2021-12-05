
import cv2
import mediapipe as mp
import time
import math

pTime=0
cTime=0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraws = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 1)

while(True):
    pTime=time.time()
    sta, frame = cap.read()
    h,w,c = frame.shape
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    pts = result.multi_hand_landmarks

    wpos=0
    hpos=0

    if pts:
        for handlms in pts:
            for id , points in enumerate(handlms.landmark):
                wpos1=int(handlms.landmark[8].x*w)
                hpos1=int(handlms.landmark[8].y*h)
                wpos2=int(handlms.landmark[4].x*w)
                hpos2=int(handlms.landmark[4].y*h)
                x=abs(wpos2-wpos1)
                y=abs(hpos2-hpos1)
                dist=math.sqrt((x**2+y**2))
                cv2.line(frame,(wpos1,hpos1),(wpos2,hpos2),(100,200,80),5)
                cv2.putText(frame,f'({int(dist)})',(wpos1+x,hpos1+y),cv2.FONT_HERSHEY_PLAIN,1.1,(20,100,255),2)

            mpDraws.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps =round(1/(cTime-pTime))
    cv2.putText(frame,f'fps: {fps}',(20,20),cv2.FONT_HERSHEY_PLAIN,1.1,(20,100,255),2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
