# Necessary modules imported
import numpy as np
import sys
import cv2

# Ensuring correct arguments
if not (len(sys.argv) == 4 or len(sys.argv) == 5):
	print "Usage : %s <dataDir> <tmplDir> <mode> {image path}" % sys.argv[0]
	sys.exit()

dataDir = sys.argv[1]
tmplDir = sys.argv[2]
mode = sys.argv[3]

eiVecs = loadModel(dataDir)
tmplData = loadTemplates(tmplDir)

img = None

if mode == "image":
	if not len(sys.argv) == 5:
		print "Usage : %s <dataDir> <tmplDir> <mode> {image path}" % sys.argv[0]
		sys.exit()
	img = cv2.imread(sys.argv[4], 0)

else if mode == "video":
	cap = cv2.VideoCapture(0)
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	while True:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)
		rois = []
		for (x, y, w, h) in faces:
			coeff = computeCoeff(gray[y:y+h, x:x+w], eiVecs)
			labelPred = np.min(map())
			cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)

		cv2.namedWindow("Cam")
		cv2.setMouseCallback("Cam", chooseFace)
		cv2.imshow("Cam", frame)
