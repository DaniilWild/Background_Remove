import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation #mediapipe
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

# imgBg = cv2.imread("images/1.jpg")
# imgBg = cv2.resize(imgBg, [640,480])

imgList = []
lmsImg = os.listdir("images")
for imgPath in lmsImg:
    img = cv2.imread("images/"+imgPath)
    img = cv2.resize(img, [640, 480])
    imgList.append(img)
print(len(imgList))
IndexImage = 0

while True:
    ret, img = cap.read()

    #imgOut = segmentor.removeBG(img,(255,0,0), threshold = 0.8 )
    imgOut = segmentor.removeBG(img, imgList[IndexImage], threshold=0.7)
    fps, imgOut = fpsReader.update(imgOut, color=(0, 255, 0))
    imgStacked = cvzone.stackImages([img, imgOut], 2,1)


    cv2.imshow("camera", imgStacked)
    #cv2.imshow("Out", imgOut)
    key  = cv2.waitKey(1)

    if key == ord("a"):
        IndexImage -=1
    elif key == ord("d"):
        IndexImage += 1
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()