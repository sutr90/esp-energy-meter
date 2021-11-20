# https://github.com/hnasr/python-on-the-backend/blob/lecture18-bonus/index.py

import tornado.web
import tornado.ioloop

import cv2
import numpy as np
import pytesseract

def parse_voda(src):
    src = src[260:380, 150:770] # 144.261 610x120
    blue_channel = src[:,:,0]
    blurred = cv2.GaussianBlur(blue_channel, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    numbers = [] # 60x100
    numbers.append(closing[5:120, 0:70])
    numbers.append(closing[5:120, 95:165])
    numbers.append(closing[5:120, 195:265])
    numbers.append(closing[5:120, 290:360])
    numbers.append(closing[5:120, 380:450])
    numbers.append(closing[5:120, 460:530])
    numbers.append(closing[5:120, 550:620])

    digits = []
    for number in numbers:
        (contours, hierarchy) = cv2.findContours(image=number, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = number.copy()
        contours = list(contours)
        m,i = max((cv2.contourArea(v),i) for i,v in enumerate(contours))
        _ = cv2.drawContours(image=image_copy, contours=contours, contourIdx=i, color=(255,255,255), thickness=cv2.FILLED)
        del contours[i]
        _ = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0,0,0), thickness=cv2.FILLED)
        img = cv2.bitwise_not(image_copy)
        digits.append(img)
    numpy_horizontal_concat = np.concatenate(digits, axis=1)
    word = pytesseract.image_to_string(numpy_horizontal_concat, config='--psm 13')

    return word

def parse_plyn(src):
    src = src[330:460, 70:800] # 70x330 730x130

    (h, w) = src.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -1.6, 1.0)
    src = cv2.warpAffine(src, M, (w, h))

    blue_channel = src[:,:,0]

    no_spec=blue_channel.copy()
    no_spec[no_spec > 190] = 0

    blurred = cv2.GaussianBlur(no_spec, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, -15)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    numbers = [] # 60x100
    numbers.append(opening[15:115, 30:90])
    numbers.append(opening[15:115, 115:175])
    numbers.append(opening[15:115, 195:255])
    numbers.append(opening[15:115, 285:345])
    numbers.append(opening[15:115, 370:430])
    numbers.append(opening[15:115, 455:515])
    numbers.append(opening[15:115, 545:605])
    numbers.append(opening[15:115, 650:710])


    digits = []
    for number in numbers:
        (contours, hierarchy) = cv2.findContours(image=number, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_copy = number.copy()
        contours = list(contours)
        
        m,i = max((cv2.contourArea(v),i) for i,v in enumerate(contours))
        _ = cv2.drawContours(image=image_copy, contours=contours, contourIdx=i, color=(255, 255,255), thickness=cv2.FILLED)
        del contours[i]
        _ = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED)
        img = cv2.bitwise_not(image_copy)
        digits.append(img)

    numpy_horizontal_concat = np.concatenate(digits, axis=1)

    word = pytesseract.image_to_string(numpy_horizontal_concat, config='--psm 8')
    return word

class index(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class uploadImgHandler_voda(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["imageFile"]
        for f in files:
            fh = open("upload/voda.jpg", "wb")
            fh.write(f.body)
            fh.close()

        nparr = np.frombuffer(f.body, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        text = parse_voda(img_np)

        self.write(text)

class uploadImgHandler_plyn(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["imageFile"]
        for f in files:
            fh = open("upload/plyn.jpg", "wb")
            fh.write(f.body)
            fh.close()

        nparr = np.frombuffer(f.body, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        text = parse_plyn(img_np)

        self.write(text)

if (__name__ == "__main__"):
    app = tornado.web.Application([
        ("/", index),
        ("/plyn", uploadImgHandler_plyn),
        ("/voda", uploadImgHandler_voda),
        ("/img/(.*)", tornado.web.StaticFileHandler, {'path': 'upload'})
    ])

    app.listen(8080)
    print("Listening on port 8080")
    tornado.ioloop.IOLoop.instance().start()