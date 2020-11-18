from imutils.object_detection import non_max_suppression
import imutils
import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cam = cv2.VideoCapture("videos/people.mp4")

while True:

    rects, image = cam.read()
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.1)

    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=5)

    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("[INFO]: {} caixas originais, {} apos supressaão".format(len(rects), len(pick)))
    
    cv2.imshow("Antes NMS", orig)
    cv2.imshow("Apos NMS", image)
    k = cv2.waitKey(60)
    if k == 27:
        break

cv2.destroyAllWindows()
