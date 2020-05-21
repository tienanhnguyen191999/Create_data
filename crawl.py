import cv2
import urllib.request
import keras_ocr
import os
import time


for i in range(1000):
    print(f'___________{i}___________')
    capchaUrl = 'http://tracuuhoadon.gdt.gov.vn/Captcha.jpg'
    urllib.request.urlretrieve(capchaUrl, "tmp/capcha.jpg")

    img = cv2.imread('tmp/capcha.jpg')
    w,h,c = img.shape
    img = cv2.resize(img,(int(128 * h/w),128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    threshold = cv2.threshold(gray,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    for contour in contours:
        if cv2.contourArea(contour) > 80:
            px, py, wx, wy = cv2.boundingRect(contour)
            cv2.rectangle(img,(px,py),(px+wx, py+wy),(0,0,255),1)

            pipline = keras_ocr.pipeline.Pipeline()
            image = keras_ocr.tools.read(img[py:py+wy, px:px+wx])
            predict = pipline.recognize([image])
            if len(predict[0]) != 0:
                filePath = f'data/{predict[0][0][0]}'
                if not os.path.exists(filePath):
                    os.mkdir(filePath)
                fileName = f'{predict[0][0][0]}_{str(time.time())}.jpg'
                cv2.imwrite(os.path.join(filePath, fileName), img[py:py+wy, px:px+wx])
