import cv2

video = cv2.VideoCapture("videos/people.mp4")
cascadePeople = cv2.CascadeClassifier('recog/haarcascade_pedestrian.xml')

while True:
    _, frame = video.read()
    peopleCinza = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    peopleRecog = cascadePeople.detectMultiScale(peopleCinza, 1.8, 7)
    for (x,y,w,h) in peopleRecog:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("People Recognition", frame)
    key = cv2.waitKey(60)
    if key == 27:
        break
cv2.destroyAllWindows()