# Necessary modules imported
import numpy as np
import sys
import cv2

# Ensuring correct arguments
if not len(sys.argv) == 3:
	print "Usage : %s <trainFile> <dataDir>" % sys.argv[0]
	sys.exit()

trainFile = sys.argv[1]
dataDir = sys.argv[2]

trainList = loadListFromFile(trainFile)
dim, mean, eiVals, eiVecs = train(trainList)

saveModel(dataDir, eiVecs)
