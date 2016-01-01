# Necessary modules imported
import numpy as np
import sys
import cv2
from EigenFace import *

# Ensuring correct arguments
if not (len(sys.argv) == 4 or len(sys.argv) == 5):
	print "Usage : %s <dataDir> <tmplDir> <mode> {image path}" % sys.argv[0]
	sys.exit()

curDir = os.getcwd() + os.sep
dataDir = curDir + sys.argv[1]
tmplDir = curDir + sys.argv[2]
mode = sys.argv[3]

if not (mode == "image" or mode == "video"):
	print "Invalid mode : %s. <mode> can be [image] or [video]" % mode
	sys.exit()

mean, eiVecs = loadModel(dataDir)
tmplData = loadTemplates(tmplDir)

def predict(img):
	coeffPred = computeCoeff(img, mean, eiVecs)
	temp = dict(map(lambda (k, v): (k, computeLoss(v, coeffPred)), tmplData.iteritems()))
	labelPred = min(temp, key=temp.get)
	return labelPred, temp[labelPred]

img = None

if mode == "image":
	if not len(sys.argv) == 5:
		print "Usage : %s <dataDir> <tmplDir> <mode> {image path}" % sys.argv[0]
		sys.exit()
	img = cv2.imread(sys.argv[4], 0)
	labelPred, loss = predict(img)
	print "Recognised :", labelPred, "Loss :", str(loss)

elif mode == "video":
	cap = cv2.VideoCapture(0)
	faceCascade = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml')
	while True:
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)
		rois = []
		for (x, y, w, h) in faces:
			labelPred, loss = predict(gray[y:y+h, x:x+w])
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(frame, labelPred + ', ' + str(loss), (x+w/2,y-10), font, 0.6, (255, 255, 255), 1)
			cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)

		k = cv2.waitKey(1)
		if k == 27:
			break

		cv2.namedWindow("Cam")
		cv2.imshow("Cam", frame)
