__author__ = 'Vamshi'

import sys
import os
import scipy.io.wavfile
import numpy as np
from scikits.talkbox.features import mfcc

genres = {'classical': 0, 'country': 1, 'jazz': 2, 'metal': 3, 'pop': 4, 'rock': 5}
no_of_docs = 600
no_of_fft_features = 1000
no_of_mfcc_features = 13


def getfftdata(path):
    classesmatrix = np.zeros((no_of_docs, 1))
    fftdata = np.zeros((no_of_docs, no_of_fft_features))
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
    np.savetxt('classesmatrixfft.txt', classesmatrix, '%d')
    return fftdata


def getmfccdata(path):
    classesmatrix = np.zeros((no_of_docs, 1))
    mfccdata = np.zeros((no_of_docs, no_of_mfcc_features))
    fileindex = 0
    for subdir, dirs, files in os.walk(path):
        if os.path.basename(subdir) in genres.keys():
            for f in files:
                if f.endswith('.wav'):
                    print "Processing file : " + f
                    sample_rate, X = scipy.io.wavfile.read(os.path.join(subdir, f))
                    ceps, mspec, spec = mfcc(X)
                    num_ceps = ceps.shape[0]
                    mfcc_features = np.mean( ceps[int( num_ceps * 1 / 10 ):int( num_ceps * 9 / 10 )] , axis=0 )
                    for i in range(len(mfcc_features)):
                        mfccdata[fileindex][i] = mfcc_features[i]
                    classesmatrix[fileindex] = genres[os.path.basename(subdir)]
                    fileindex += 1
    np.savetxt('classesmatrixmfcc.txt', classesmatrix, '%d')
    return mfccdata


# Split the fft data into train and test data. Generates 10 sets of data.
def kfold(fftdata, numberoffolds):
    folddata = []
    classesmatrix = np.loadtxt('classesmatrixfft.txt', int, '%d')
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


# Load the fft data from fft data file
def loadfftdata(file):
    return np.loadtxt(file, float, '%f')


# Saves the data into a txt file.
def savemfccdatatofile(mfccdata):
    np.savetxt('mfccdata.txt', mfccdata, '%f')


# Load the fft data from mfcc data file
def loadmfccdata(file):
    return np.loadtxt(file, float, '%f')


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
    np.savetxt('test.txt',newtestclasses,'%d')
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


def processdata(fftdata, no_of_features):
    folddata = kfold(fftdata, 10) # 10-fold cross validation
    eta = 0.01
    eta_new = 0.01
    lmda = 0.001
    it = 3000
    eachfoldmaxaccuracies = []
    for i in range(len(folddata)):
        # if i == 1:
            weights = np.zeros((len(genres), no_of_features + 1))
            train, test, testclasses = folddata[i]
            train = normalize(train)
            test = normalize(test)
            tempweights = weights[:]
            maxaccuracy = 0
            for j in range(it):
                # if j == 0 or j == 1:
                    print "Current Fold : " + str(i)
                    print "Iteration : " + str(j)
                    eta = eta_new / (1 + float(j) / it)
                    tempweights = trainfn(train, tempweights, eta, lmda)
                    # np.savetxt('c:/temp/weights.txt', tempweights, '%f')
                    newtestclasses = testfn(tempweights, test)
                    confmatrix, accuracy = calc_conf_acc(testclasses, newtestclasses)
                    if accuracy > maxaccuracy:
                        maxaccuracy = accuracy
                    print "Accuracy  : " + str(accuracy)
                    print "Confusion Matrix : \n" + str(confmatrix)
            eachfoldmaxaccuracies.append(maxaccuracy)
    for i in range(len(eachfoldmaxaccuracies)):
        print "Iteration " + str(i) + " max accuracy : " + str(eachfoldmaxaccuracies[i])
    print "Avg of all folds accuracies : " + str(np.average(eachfoldmaxaccuracies))


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
                        processdata(loadfftdata("fftdata.txt"), no_of_fft_features)
                        exit(0)
                    else:
                        print "fftdata.txt is not present. Creating one."
                        savefftdatatofile(getfftdata(args[1]))
                        processdata(loadfftdata("fftdata.txt"), no_of_fft_features)
                        exit(0)
                elif args[0] == "-mfcc":
                    print "Using mfcc..."
                    print "Checking if mfcc data file (mfccdata.txt) is already present..."
                    if os.path.isfile('mfccdata.txt'):
                        print "mfccdata.txt is already present. Using this file."
                        processdata(loadmfccdata("mfccdata.txt"), no_of_mfcc_features)
                        exit(0)
                    else:
                        print "mfccdata.txt is not present. Creating one."
                        savemfccdatatofile(getmfccdata(args[1]))
                        processdata(loadmfccdata("mfccdata.txt"), no_of_mfcc_features)
                        exit(0)
            else:
                print "Invalid arguments. Usage: -fft/mfcc <path to data>"
                exit(0)