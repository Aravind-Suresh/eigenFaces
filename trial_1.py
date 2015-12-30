# Necessary modules imported
import numpy as np
import sys
import cv2

# Ensuring correct arguments
if not len(sys.argv) == 4:
	print "Usage : %s <trainFile> <testFile> <outputDirectory>" % sys.argv[0]
	sys.exit()

# Constants
dim = cv2.imread(trainList[0], 0).shape

def loadListFromFile(filePath):
	return map(lambda path: path.strip(), open(filePath, 'r').readlines())

def train(trainList):
	trainset = [ np.ravel(cv2.imread(path, 0)) for path in trainList ]
	N = len(trainset)

	# Computing mean image
	mean = np.mean(trainset, 0)
	# print N, np.array(trainset).shape, mean.shape

	cov = lambda mat: np.dot(np.reshape(mat-mean, (-1, 1)), np.reshape(mat-mean, (1, -1)))
	C = (1.0/N)*np.array(np.sum((map(cov, trainset)), 0))

	w, v = np.linalg.eig(C)

	return mean, w, v

def test(testList, eiVecs):
	temp = map(lambda line: line.split(' '), testList)
	testset = [ np.ravel(cv2.resize(cv2.imread(path, 0), dim)) for path in temp[:, 0]]
	testLabels = temp[:, 1]

	output = [ compute(img, eiVecs) for img in testset ]

	# TODO : Compute loss and give a match
	return output

def compute(img, eiVecs):
	coeff = map(lambda v: np.dot(v, imgR), eiVecs)
	return coeff

trainFile = sys.argv[1]
testFile = sys.argv[2]
outputDirectory = sys.argv[3]

trainList = loadListFromFile(trainFile)
testList = loadListFromFile(testFile)
dim, mean, eiVals, eiVecs = train(trainList)

img = cv2.imread(testFile, 0)
coeff = compute(img, dim, eiVecs)
