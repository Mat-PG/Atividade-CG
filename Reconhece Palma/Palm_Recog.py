import cv2

camera = cv2.VideoCapture(0)
cascadePalm = cv2.CascadeClassifier('recog/palm_v4.xml')

while True:
    _, frame = camera.read()
    palmCinza = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    palmRecog = cascadePalm.detectMultiScale(palmCinza, 1.1, 5)
    for (x,y,w,h) in palmRecog:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(60)
    if key == 27:
        break
cv2.destroyAllWindows()
camera.release()