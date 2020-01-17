import cv2

class Haar(object):
    
    
    """use opencv and Haar Cascade for face detection"""
    def face_detect(img, net):

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = net.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
        except Exception as e:
            print("failed HAAR" + str(e))
        return img