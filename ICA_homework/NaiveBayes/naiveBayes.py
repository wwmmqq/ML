__author__= "wmq"
import string
from numpy import *


def loadDataSet(filepath):#'spam_train.txt'
	flag = []
	emails = []
	f = open(filepath, 'r')

	for line in f:
		flag.append( string.atoi(line[0]) )
		temp = line[2:].split()
		emails.append([token for token in temp if len(token) > 1]) # tolower()
	f.close()
	
	return emails, flag


# build our words dictionary
def makeWordDict(emails):
	wordsDict = set([])
	for oneEmailVector in emails:
		wordsDict = wordsDict | set(oneEmailVector)
	return list(wordsDict)



# translate a email text to a vector
def emailWords2vec(wordsDict, inputAnEmailVector):

	vec =[0] * len(wordsDict)
	for token in inputAnEmailVector:
		if token in wordsDict:
			vec[wordsDict.index(token)] = 1
		else:
			pass
	return vec


def trainNaiveBayes(EmailMatrix, flag):
	numOfEmails = len(EmailMatrix)
	numWords = len(EmailMatrix[0])
	p1 = sum(flag) / float(numOfEmails)
	p0 = 1 - p1

	## Laplace smoothing
	pw0_molecular = ones(numWords)
	pw1_molecular = ones(numWords)
	p0_molecular = 2.0	# because s can be 1 or 0
	p1_molecular = 2.0

	for i in range(numOfEmails):
		if flag[i] == 1:
			pw1_molecular += EmailMatrix[i]
			p1_molecular += sum(EmailMatrix[i])
		else:
			pw0_molecular += EmailMatrix[i]
			p0_molecular += sum(EmailMatrix[i])

	pw0 = log(pw0_molecular / p0_molecular)
	pw1 = log(pw1_molecular / p1_molecular)

	return pw0, pw1, p0, p1

# use log to replace *, to void it too samll.
def calssify(p0, p1, pw0, pw1, anEmailVector):
	p0 = sum(pw0 * anEmailVector)  + log(p0)
	p1 = sum(pw1 * anEmailVector) + log(p1)

	if p0 > p1:
		return 0
	else:
		return 1

def testNaiveBayes(p0, p1, pw0, pw1, wordsDict):
	testEmails, flag = loadDataSet('spam_test.txt')
	testEmailMatrix = []
	for i in range(len(testEmails)):
		testEmailMatrix.append(emailWords2vec(wordsDict, testEmails[i]))

	errorNum  = 0
	for i in range(len(testEmailMatrix)):
		if calssify(p0, p1, pw0, pw1, testEmailMatrix[i]) != flag[i]:
			errorNum += 1

	print "error rate is : "
	print errorNum / float(len(testEmails))
	print "error count :  "
	print errorNum


def main():
	emails, flag = loadDataSet('spam_train.txt')
	wordsDict = makeWordDict(emails)
	trainEmailMatrix = []
	for i in range(len(emails)):
		trainEmailMatrix.append(emailWords2vec(wordsDict, emails[i]))

	#print "hahahhahah"
	pw0, pw1, p0, p1 = trainNaiveBayes(trainEmailMatrix, flag)

	print "p0 = "
	print p0
	print "trainNaiveBayes is ok !!!!!!!"
	#print p0
	#print p1
	testNaiveBayes(p0, p1, pw0, pw1, wordsDict)

	# error rate is : 0.017

if __name__ == '__main__':
	main()

