# Necessary modules imported
import numpy as np
import sys
import cv2

# Ensuring correct arguments
if not len(sys.argv) == 6:
	print "Usage : %s <dataDir> <tmplDir> <label> <mode> {image path}" % sys.argv[0]
	sys.exit()

dataDir = sys.argv[1]
tmplDir = sys.argv[2]
label = sys.argv[3]
mode = sys.argv[4]

eiVecs = loadModel(dataDir)

img = None

if mode == "image":
	if not len(sys.argv) == 6:
		print "Usage : %s <dataDir> <tmplDir> <label> <mode> {image path}" % sys.argv[0]
		sys.exit()
	img = cv2.imread(sys.argv[5], 0)

else if mode == "video":
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		k = cv2.waitKey(1)
		if k == 27:
			# Choose rectangle, if no rectangle abort
			break
		cv2.imshow("Cam", frame)

coeff = computeCoeff(img, eiVecs)
saveTemplate(tmplDir, label, coeff)
