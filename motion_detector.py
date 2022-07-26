import cv2, time
import pandas as pd
from datetime import datetime 

first_frame = None
status_list = [None, None]
times = []

df = pd.DataFrame(columns = ["Start", "End"])

# video = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
video = cv2.VideoCapture(0)     #if you have f.e 2 cameras you can enetr 0 or 1 etc.

# a = 0 to check how many frames we have in video

while True:
    # a = a+1
    check, frame = video.read()
    status = 0
    # print(check)
    # print(frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    thresh_frame = cv2.threshold(delta_frame, 30 ,255, cv2.THRESH_BINARY)[1] # [1] means we want to access second item of the tuple

    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        status = 1
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # time.sleep(1)
    cv2.imshow("Gray Frame", gray)   #gray or frame
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color frame", frame)

    key = cv2.waitKey(1)
    # print(gray)
    # print(delta_frame)

    if key == ord('o'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0, len(times),2):  # step == 2
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)
# print(a)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows