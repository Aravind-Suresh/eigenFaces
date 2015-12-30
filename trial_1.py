# Necessary modules imported
import numpy as np
import sys
import cv2

# Ensuring correct arguments
if not len(sys.argv) == 3:
	print "Usage : %s <trainFile> <testPath>" % sys.argv[0]
	sys.exit()

def loadListFromFile(filePath):
	return map(lambda path: path.strip(), open(filePath, 'r').readlines())

def train(trainList):
	trainset = [ np.ravel(cv2.imread(path, 0)) for path in trainList ]
	N = len(trainset)
	dim = cv2.imread(trainList[0], 0).shape

	# Computing mean image
	mean = np.mean(trainset, 0)
	# print N, np.array(trainset).shape, mean.shape

	cov = lambda mat: np.dot(np.reshape(mat-mean, (-1, 1)), np.reshape(mat-mean, (1, -1)))
	C = (1.0/N)*np.array(np.sum((map(cov, trainset)), 0))

	w, v = np.linalg.eig(C)

	return dim, mean, w, v

def compute(img, dim, eiVecs):
	imgR = cv2.resize(img, dim).ravel()
	coeff = map(lambda v: np.dot(v, imgR), eiVecs)
	return coeff

trainFile = sys.argv[1]
testPath = sys.argv[2]

trainList = loadListFromFile(trainFile)
dim, mean, eiVals, eiVecs = train(trainList)

img = cv2.imread(testPath, 0)
coeff = compute(img, dim, eiVecs)

print coeff
