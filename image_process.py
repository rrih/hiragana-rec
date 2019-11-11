import cv2

def imwrite(image):
    mylist = []
    mylist.append(image [0:400, 0:400])
    mylist.append(image [0:400, 400:800])
    mylist.append(image [0:400, 800:1200])
    mylist.append(image [0:400, 1200:1600])
    mylist.append(image [0:400, 1600:2000])
    return mylist
    