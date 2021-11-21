# https://github.com/hnasr/python-on-the-backend/blob/lecture18-bonus/index.py

import tornado.web
import tornado.ioloop

import cv2
import numpy as np
import pytesseract
import re

def parse_voda(src):
    src = src[200:320, 40:560] # 40,200 520x120 # y1:y2, x1,x2
    blue_channel = src[:,:,0]
    blurred = cv2.GaussianBlur(blue_channel, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    numbers = [] # 60x100
    numbers.append(closing[5:120, 0:60])
    numbers.append(closing[5:120, 95:155])
    numbers.append(closing[5:120, 185:245])
    numbers.append(closing[5:120, 260:320])
    numbers.append(closing[5:120, 330:390])
    numbers.append(closing[5:120, 395:455])
    numbers.append(closing[5:120, 480:520])

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
    persist_image(numpy_horizontal_concat, 'voda')
    word = pytesseract.image_to_string(numpy_horizontal_concat, config='--psm 13')
    return re.sub('[\W]', '', word)

def parse_plyn(src):
    src = src[210:330, 25:575] # 25,210 550x120 # y1:y2, x1:x2

    blue_channel = src[:,:,0]

    no_spec=blue_channel.copy()

    blurred = cv2.GaussianBlur(no_spec, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, -15)
    
    (contours, hierarchy) = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    for i,v in enumerate(contours):
        x,y,w,h = cv2.boundingRect(v)
        if w > 50:
             _ = cv2.drawContours(image=thresh, contours=contours, contourIdx=i, color=(0,0,0), thickness=cv2.FILLED)

    kernel = np.ones((5,1),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, np.ones((1,5),np.uint8))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_ERODE, np.ones((1,5),np.uint8))

    numbers = [] # 50x100
    numbers.append(opening[15:115, 10:60])
    numbers.append(opening[15:115, 90:140])
    numbers.append(opening[15:115, 170:220])
    numbers.append(opening[15:115, 245:295])
    numbers.append(opening[15:115, 320:370])
    numbers.append(opening[15:115, 395:445])
    numbers.append(opening[15:115, 490:540])


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
    persist_image(numpy_horizontal_concat, 'plyn')

    word = pytesseract.image_to_string(numpy_horizontal_concat, config='--psm 13')
    return re.sub('[\W]', '', word)

def persist_image(image, label):
    timestamp = datetime.timestamp(now)
    filename = '{}-{}.png'.format(timestamp, label)
    cv2.imwrite('data/' + filename, image)

def persist_data(filename, data):
    with open(filename, "a") as file_object:
        file_object.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        file_object.write(" ")
        file_object.write(data)
        file_object.write("\n")

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
        persist_data("voda.txt", text)

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
        persist_data("plyn.txt", text)

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