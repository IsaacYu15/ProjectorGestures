import requests
import cv2
from threading import Timer
import mediapipe as mp
import numpy as np
import pyautogui
import autopy
import time
import speech_recognition as sr

class GestureRecognizer:
    def __init__(self, activeMode=False, maxHands=1, detectionConfidence=False, trackingConfidence=0.5):
        self.activeMode = activeMode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mediaPipeHands = mp.solutions.hands
        self.handProcessor = self.mediaPipeHands.Hands(self.activeMode, self.maxHands, self.detectionConfidence, self.trackingConfidence)
        self.mediaPipeDrawing = mp.solutions.drawing_utils
        self.fingerTipIndices = [4, 8, 12, 16, 20]

    def detectHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.processResults = self.handProcessor.process(frameRGB)

        if self.processResults.multi_hand_landmarks:
            for handLandmarks in self.processResults.multi_hand_landmarks:
                if draw:
                    self.mediaPipeDrawing.draw_landmarks(frame, handLandmarks, self.mediaPipeHands.HAND_CONNECTIONS)
        return frame

    def getPositions(self, frame, handIndex=0, draw=True):   
        coordXList, coordYList, boundingBox, self.landmarkList  = [], [], [], []

        if self.processResults.multi_hand_landmarks:
            selectedHand = self.processResults.multi_hand_landmarks[handIndex]
            
            for idx, landmark in enumerate(selectedHand.landmark):
                height, width, _ = frame.shape
                coordX, coordY = int(landmark.x * width), int(landmark.y * height)
                coordXList.append(coordX)
                coordYList.append(coordY)
                self.landmarkList.append([idx, coordX, coordY])
                if draw:
                    cv2.circle(frame, (coordX, coordY), 5, (255, 0, 255), cv2.FILLED)

            xMin, xMax = min(coordXList), max(coordXList)
            yMin, yMax = min(coordYList), max(coordYList)
            boundingBox = xMin, yMin, xMax, yMax

            if draw:
                cv2.rectangle(frame, (xMin - 20, yMin - 20), (xMax + 20, yMax + 20),
                              (0, 255, 0), 2)

        return self.landmarkList, boundingBox

    def pinching(self):
        dist = (pow(self.landmarkList[self.fingerTipIndices[0]][1] - self.landmarkList[self.fingerTipIndices[1]][1], 2) +
                pow(self.landmarkList[self.fingerTipIndices[0]][0] - self.landmarkList[self.fingerTipIndices[1]][0], 2) )
        
        return dist

def empty_callback(*args, **kwargs):
    pass

def reset():
    url = 'http://localhost:80/elm/groups/Group01/performer?active=1&sequenceId=17'
    requests.post(url)

def on_click():
    # autopy.key.tap(autopy.key.Code.DOWN_ARROW)
    autopy.mouse.click()
    url = 'http://localhost:80/elm/groups/Group01/performer?active=1&sequenceId=19'
    requests.post(url)

    t = Timer(1, reset)
    t.start()

def main():
    w = 600
    h = 480
    edgeBuffer = 10
    smooth = 8
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    prev_time = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, w) 
    cap.set(4, h)  
    cap.set(cv2.CAP_PROP_FPS, 60) 

    detector = GestureRecognizer()               
    scr_w, scr_h = autopy.screen.size()
    scr_w -= edgeBuffer
    scr_h -= edgeBuffer

    click = False

    buffer_max = 5
    state_buffer = [] #0 move, 1 drag

    cv2.namedWindow("Archand", cv2.WINDOW_NORMAL)
    cv2.createTrackbar('W','Archand',100, round(w / 2), empty_callback)
    cv2.createTrackbar('H','Archand',100, round(h / 2), empty_callback)

    reset()
    
    while True:
        _, img = cap.read()
        process_frame = cv2.flip(img, 1) 
        img = detector.detectHands(process_frame)
        lmList, _ = detector.getPositions(process_frame)           

        frameW = cv2.getTrackbarPos('W','Archand')
        frameH = cv2.getTrackbarPos('H','Archand')

        leftBound = round(w/2 - frameW)
        rightBound = round(w / 2 + frameW)
        topBound = round(h/2-frameH)
        botBound = round(h/2 + frameH)
        
        cv2.rectangle(img, (leftBound, topBound) , (rightBound, botBound), (255, 0, 255), 2)

        if len(lmList) != 0:
            x1 = round ((lmList[detector.fingerTipIndices[0]][1] + lmList[detector.fingerTipIndices[1]][1])/2)
            y1 = round ((lmList[detector.fingerTipIndices[1]][2] + lmList[detector.fingerTipIndices[1]][2])/2)

            pinch = detector.pinching()
            if (pinch < 100):
                state_buffer.append(1)
            else:    
                state_buffer.append(0)
            if len(state_buffer) > buffer_max:
                state_buffer.pop(0)

            x3 = np.interp(x1, (leftBound, rightBound), (edgeBuffer, scr_w))
            y3 = np.interp(y1, (topBound, botBound), (edgeBuffer, scr_h))

            if ( not (x3 <= edgeBuffer or x3 >= scr_w - edgeBuffer or 
                y3 <= edgeBuffer or y3 >= scr_h - edgeBuffer) ):

                dragging = sum(state_buffer) / len(state_buffer) > 0.5

                if dragging:
                    if not click:
                        on_click()
                        click = True
                else:
                    click = False

                    curr_x = prev_x + (x3 - prev_x) / smooth
                    curr_y = prev_y + (y3 - prev_y) / smooth
                    autopy.mouse.move(scr_w - curr_x, curr_y)

                    cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                    prev_x, prev_y = curr_x, curr_y

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        cv2.imshow("Archand", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()