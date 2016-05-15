#!/usr/bin/env python

import cv2
import sys
import glob

# https://realpython.com/blog/python/face-recognition-with-python/

#imagePath = sys.argv[1]
#imagePath = '/Pictures/20150917_045611.jpg'
imageGlob = glob.glob('/Pictures/*jpg')

cascPath = '/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for imgPath in imageGlob:
    image = cv2.imread(imagePath)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
        #flags = cv2.CASCADE_SCALE_IMAGE
        flags = 0
    )
    print "Found {0} faces!".format(len(faces))


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
