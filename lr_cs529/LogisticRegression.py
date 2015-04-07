__author__ = 'Vamshi'

import sys
import os
import scipy.io.wavfile
import numpy as np
from scikits.talkbox.features import mfcc

genres = {'classical': 0, 'jazz': 1, 'country': 2, 'pop': 3, 'rock': 4, 'metal': 5}
no_of_docs = 600
no_of_features = 1000

def getdata(path):
    classesmatrix = np.zeros((no_of_docs, 1))
    fftdata = np.zeros((no_of_docs, no_of_features))
    fileindex = 0
    for subdir, dirs, files in os.walk(path):
        if os.path.basename(subdir) in genres.keys():
            for f in files:
                if f.endswith('.wav'):
                    print "Processing file : " + f
                    sample_rate, X = scipy.io.wavfile.read(os.path.join(subdir, f))
                    fft_features = abs(scipy.fft(X)[:1000])
                    for i in range(len(fft_features)):
                        fftdata[fileindex][i] = fft_features[i]
                    classesmatrix[fileindex] = genres[os.path.basename(subdir)]
                    fileindex += 1
    np.savetxt('classesmatrix.txt', classesmatrix, '%d')
    return fftdata


# Split the fft data into train and test data. Generates 10 sets of data.
def kfold(fftdata, numberoffolds):
    folddata = []
    classesmatrix = np.loadtxt('classesmatrix.txt', int, '%d')
    for i in range(numberoffolds):
        train = []
        test = []
        testclasses = []
        for j in range(len(fftdata)):
            if (j - i) % 10 == 0:
                test.append(fftdata[j])
                testclasses.append(classesmatrix[j])  # append 0,10,20,...
            else:
                train.append(fftdata[j])
        folddata.append((train, test, testclasses))
    return folddata


# Saves the data into a txt file.
def savefftdatatofile(fftdata):
    np.savetxt('fftdata.txt', fftdata, '%f')


# Load the fft data from pickled fft data file 'pickledfftdata.pkl'
def loadfftdata(picklefile):
    return np.loadtxt('fftdata.txt', float, '%f')


# Normalize the data (both train, test)
def normalize(data):
    for i in range(len(data[0])):
        maxv = 0
        for j in range(len(data)):
            if data[j][i] > maxv:
                maxv = data[j][i]
        for j in range(len(data)):
            data[j][i] = data[j][i]/maxv
    for i in range(len(data)):
        data[i] = np.concatenate([[1], data[i]])
    return data


def trainfn(train, tempweights, eta, lmda):
    deltamatrix = np.zeros((len(genres), len(train)))
    count = 0
    for i in range(len(genres)):
        for j in range(len(train)):
            if j >= count and j < count + 90:
                deltamatrix[i][j] = 1
        count += 90
    p = np.exp(tempweights.dot(np.transpose(train)))
    for i in range(len(p[0])):
        p[len(p)-1][i] = 1
    for i in range(len(p[0])):
        p[:, i] = p[:, i]/np.sum(p[:, i])

    errormatrix = (deltamatrix - p).dot(train)

    intermatrix = eta * (errormatrix - lmda * tempweights)

    updatedweightmatrix = np.add(tempweights, intermatrix)

    return updatedweightmatrix


def testfn(tempweights, test):
    p = np.exp(tempweights.dot(np.transpose(test)))

    for i in range(len(p[0])):
        p[len(p)-1][i] = 1
    for i in range(len(p[0])):
        p[:, i] = p[:, i]/np.sum(p[:, i])

    newtestclasses = np.zeros((len(test), 1))
    for i in range(len(p[0])):
        maxv = 0
        for j in range(len(p)):
            if p[j][i] > maxv:
                maxv = p[j][i]
                maxj = j
        newtestclasses[i] = maxj

    return newtestclasses


def calc_conf_acc(testclasses, newtestclasses):
    confusionmatrix = np.zeros((len(genres), len(genres)))
    correct = 0
    for i in range(len(testclasses)):
        o = int(testclasses[i]) - 1
        n = int(newtestclasses[i]) - 1
        confusionmatrix[o][n] += 1
        if testclasses[i] == newtestclasses[i]:
            correct += 1
    return confusionmatrix, float(correct)/len(testclasses)


def processfftdata(fftdata):
    folddata = kfold(fftdata, 10)
    eta = 0.01
    eta_new = 0.01
    lmda = 0.001
    it = 300
    for i in range(len(folddata)):
            weights = np.zeros((len(genres), no_of_features + 1))
            train, test, testclasses = folddata[i]
            train = normalize(train)
            test = normalize(test)
            tempweights = weights[:]
            for j in range(it):
                    print "Current Fold : " + str(i)
                    print "Iteration : " + str(j)
                    eta = eta_new / (1 + j / it)
                    tempweights = trainfn(train, tempweights, eta, lmda)
                    newtestclasses = testfn(tempweights, test)
                    confmatrix, accuracy = calc_conf_acc(testclasses, newtestclasses)
                    print "Accuracy  : " + str(accuracy)
                    print "Confusion Matrix : \n" + str(confmatrix)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Not enough arguments. Usage: -fft/-mfcc <path to data>"
        exit(0)
    else:
        if os.path.isdir(args[1]):
            if args[0] == "-fft" or args[0] == "-mfcc":
                if args[0] == "-fft":
                    print "Using fft..."
                    print "Checking if fft data file (fftdata.txt) is already present..."
                    if os.path.isfile('fftdata.txt'):
                        print "fftdata.txt is already present. Using this file."
                        processfftdata(loadfftdata("fftdata.txt"))
                        exit(0)
                    else:
                        print "Pickledfftdata.pkl is not present. Creating one."
                        savefftdatatofile(getdata(args[1]))
                        processfftdata(loadfftdata("fftdata.txt"))
                        exit(0)
                elif args[0] == "-mfcc":
                    print "Using mfcc..."
                else:
                    print "Incorrect arguments. Usage: -fft/mfcc <path to data>"
                    exit(0)