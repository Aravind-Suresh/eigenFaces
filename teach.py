# Necessary modules imported
import numpy as np
import sys
import cv2
from EigenFace import *

# Ensuring correct arguments
if not (len(sys.argv) == 5 or len(sys.argv) == 6):
	print "Usage : %s <dataDir> <tmplDir> <label> <mode> {image path}" % sys.argv[0]
	sys.exit()

curDir = os.getcwd() + os.sep
dataDir = curDir + sys.argv[1]
tmplDir = curDir + sys.argv[2]
label = sys.argv[3]
mode = sys.argv[4]

mean, eiVecs = loadModel(dataDir)

img = None

facesG = None
faceIdx = None

def chooseFace(event, x, y, flags, param):
	global faceIdx
	if event == cv2.EVENT_LBUTTONUP:
		if not facesG == None:
			idx = 0
			for(_x, _y, _w, _h) in facesG:
				if x>_x and x<_x+_w and y>_y and y<_y+_h:
					# print idx
					faceIdx = idx
					break
				idx += 1

if mode == "image":
	if not len(sys.argv) == 6:
		print "Usage : %s <dataDir> <tmplDir> <label> <mode> {image path}" % sys.argv[0]
		sys.exit()
	img = cv2.imread(sys.argv[5], 0)

elif mode == "video":
	cap = cv2.VideoCapture(0)
	faceCascade = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml')
	while True:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)
		facesG = faces

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
		k = cv2.waitKey(1)
		if k == 27:
			break

		if not faceIdx == None:
			img = gray[y:y+h, x:x+w]
			break

		cv2.namedWindow("Cam")
		cv2.setMouseCallback("Cam", chooseFace)
		cv2.imshow("Cam", frame)

coeff = computeCoeff(img, mean, eiVecs)
saveTemplate(tmplDir, label, coeff)
