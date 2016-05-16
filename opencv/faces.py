#!/usr/bin/env python

import cv2
import sys
import glob
import os

# https://realpython.com/blog/python/face-recognition-with-python/

#imagePath = sys.argv[1]
#imagePath = '/Pictures/20150917_045611.jpg'
imageGlob = glob.glob('/Pictures/*jpg')

cascPath = '/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
#cascPath = '/opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
#cascPath = '/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml'
#cascPath = '/opencv/data/lbpcascades/lbpcascade_frontalface.xml'

faceCascade = cv2.CascadeClassifier(cascPath)
for imagePath in imageGlob:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(20, 20),
        #flags = cv2.CASCADE_SCALE_IMAGE
        #flags = 0
    )
    num_faces = len(faces)
    if num_faces:
        print "Found {0} faces!".format(len(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('/tmp/detected-%s' % os.path.basename(imagePath), image)
        


