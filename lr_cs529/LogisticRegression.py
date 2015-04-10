__author__ = 'Vamshi'

import sys
import os
import scipy.io.wavfile
import numpy as np
from scikits.talkbox.features import mfcc

genres = {'classical': 0, 'country': 1, 'jazz': 2, 'metal': 3, 'pop': 4, 'rock': 5}
no_of_docs = 600
no_of_fft_features = 1000
no_of_best_fft_features = 20
no_of_mfcc_features = 13


def getfftdata(path):
    """
    This function extracts the fft data from the wav files

    Parameters:
    -----------
    path - path to get the directory name of the songs present "E:\UNM\CS 529 - Intro to Machine Learning\Assignment 3\opihi.cs.uvic.ca\sound\genres"

    Returns:
    --------
    fftdata - fft data matrix of size (600,1000)
    """
    classesmatrix = np.zeros((no_of_docs, 1))                   # Stores the song, genre information in classesmatrix.txt file -> Line number as song index, genre
    fftdata = np.zeros((no_of_docs, no_of_fft_features))        # Matrix (600,1000) to store the fft features information of all the songs in 6 genres
    fileindex = 0                                               # to store the current offset of the song
    for subdir, dirs, files in os.walk(path):                   # Traversing all the files in 6 genres
        if os.path.basename(subdir) in genres.keys():
            for f in files:
                if f.endswith('.wav'):
                    print "Processing file : " + f
                    sample_rate, X = scipy.io.wavfile.read(os.path.join(subdir, f))
                    fft_features = abs(scipy.fft(X)[:1000])     # Extracts 1000 first FFT component features.
                    for i in range(len(fft_features)):
                        fftdata[fileindex][i] = fft_features[i]
                    classesmatrix[fileindex] = genres[os.path.basename(subdir)]     # Storing the genre of every song in a matrix.
                    fileindex += 1
    np.savetxt('classesmatrix.txt', classesmatrix, '%d')     # Writing the classesmatrix to a file.
    return fftdata


def genbestfftdata(fftdata, n):
    """
    This function extracts the total fft data based on best 'n' features per genre.

    Parameters:
    -----------
    fftdata - fft data matrix (600,1000) which is generated by the function getfftdata()
    n       - 'n' best features from 1000 features. (n=20 for our assignment)

    Returns:
    --------
    bestfftdata - fft data matrix of size (600,20) based on best 'n' features per genre
    """

    # Separating the fftdata into separate genres
    classical = fftdata[0:100]
    country = fftdata[100:200]
    jazz = fftdata[200:300]
    metal = fftdata[300:400]
    pop = fftdata[400:500]
    rock = fftdata[500:600]

    fftstddata = np.std(fftdata, axis=0)        # Calculates Standard Deviation of all the features in total fft data.

    bestfftdata = np.empty((0, n))              # Initializing empty matrix to store each genre data with best 'n' best features for that genre. The size of this matrix becomes (600,n) ultimately.

    # Generate the best genre data based on best features for that genre and append to bestfftdata matrix.
    bestfftdata = np.r_[bestfftdata, genbestfftdatapergenre(classical, n, fftstddata)]
    bestfftdata = np.r_[bestfftdata, genbestfftdatapergenre(country, n, fftstddata)]
    bestfftdata = np.r_[bestfftdata, genbestfftdatapergenre(jazz, n, fftstddata)]
    bestfftdata = np.r_[bestfftdata, genbestfftdatapergenre(metal, n, fftstddata)]
    bestfftdata = np.r_[bestfftdata, genbestfftdatapergenre(pop, n, fftstddata)]
    bestfftdata = np.r_[bestfftdata, genbestfftdatapergenre(rock, n, fftstddata)]
    return bestfftdata


def genbestfftdatapergenre(genredata, n, fftstddata):
    """
    This function extracts the best fft data based on best 'n' features per genre. Used "Standard Deviation" to get the best 'n' features per genre

    Parameters:
    -----------
    genredata  - genre fft data matrix (100,1000)
    n          - 'n' best features from 1000 features. (n=20 for our assignment)
    fftstddata - Standard Deviation of all the features in total fft data.

    Returns:
    --------
    bestfftdatapergenre - fft data matrix of size (100,20) based on best 'n' features per genre
    """

    bestfftdatapergenre = np.zeros((len(genredata), n))         # Array to store Standard Deviations of all the features per genre
    bestfftfeaturesindexes = []                                 # Array to store the indexes of best features per genre
    indexeddiffstd = []                                         # Array to store the index of the Standard Deviations of all the features per genre.
    genrestddata = np.std(genredata, axis=0)                    # Calculates Standard Deviation of all the features in genre fft data.
    diffstd = fftstddata - genrestddata                         # Take the difference of the Standard Deviations of total fft data and genre fft data
    for i in range(len(diffstd)):
        indexeddiffstd.append((diffstd[i],i))

    sorteddiffstd = sorted(indexeddiffstd, reverse=True)        # Sort the indexed Standard Deviations in 'indexeddiffstd' in decreasing order.

    for i in range(len(sorteddiffstd)):                         # Take top 20 features indexes.
        bestfftfeaturesindexes.append(sorteddiffstd[i][1])
        if i == n-1:
            break
    for i in range(len(bestfftfeaturesindexes)):                # Take the data for the 20 top features determined in the previous for loop.
        bestfftdatapergenre[:, i] = genredata[:, bestfftfeaturesindexes[i]]
    return bestfftdatapergenre


def getmfccdata(path):
    """
    This function extracts the mfcc data from the wav files

    Parameters:
    -----------
    path - path to get the directory name of the songs present "E:\UNM\CS 529 - Intro to Machine Learning\Assignment 3\opihi.cs.uvic.ca\sound\genres"

    Returns:
    --------
    mfccdata - mfcc data matrix of size (600,13)
    """
    classesmatrix = np.zeros((no_of_docs, 1))                       # Stores the song, genre information in classesmatrix.txt file -> Line number as song index, genre
    mfccdata = np.zeros((no_of_docs, no_of_mfcc_features))          # Matrix (600,13) to store the fft features information of all the songs in 6 genres
    fileindex = 0                                                   # to store the current offset of the song
    for subdir, dirs, files in os.walk(path):                       # Traversing all the files in 6 genres
        if os.path.basename(subdir) in genres.keys():
            for f in files:
                if f.endswith('.wav'):
                    print "Processing file : " + f
                    sample_rate, X = scipy.io.wavfile.read(os.path.join(subdir, f))
                    ceps, mspec, spec = mfcc(X)
                    num_ceps = ceps.shape[0]
                    mfcc_features = np.mean(ceps[int(num_ceps * 1 / 10):int(num_ceps * 9 / 10)], axis=0)   # Extracts 13 features.
                    for i in range(len(mfcc_features)):
                        mfccdata[fileindex][i] = mfcc_features[i]
                    classesmatrix[fileindex] = genres[os.path.basename(subdir)]     # Storing the genre of every song in a matrix.
                    fileindex += 1
    np.savetxt('classesmatrix.txt', classesmatrix, '%d')                        # Writing the classesmatrix to a file.
    return mfccdata


def kfold(data, numberoffolds):
    """
    This function splits the fft data into train and test data. Generates 10 sets of data.

    Parameters:
    -----------
    data            - fft/fft20/mfcc data matrix
    numberoffolds   - k-fold cross validation size

    Returns:
    --------
    folddata        - k-fold data (k sets of data)

    Notes:
    -------
    Separates every 9 fft data into 'train' and every 10th fft data into 'test'. Like this, 10 sets of data is generated for 10 folds. One such fold example is given below.
    Eg: (train, test) = ({1,2,3,...,9,11,12,...,19,21,22,....599}, {0,10,20,30,....590})
    """
    folddata = []       # Array to store fold data
    classesmatrix = np.loadtxt('classesmatrix.txt', int, '%d')       # Load classes matrix file generated during getfftdata() or getmfccdata()
    for i in range(numberoffolds):
        train = []                                                   # Store the train data
        test = []                                                    # Store the test data
        testclasses = []
        for j in range(len(data)):                                   # Generates the data each fold. Explained this in Notes above.
            if (j - i) % 10 == 0:
                test.append(data[j])
                testclasses.append(classesmatrix[j])
            else:
                train.append(data[j])
        folddata.append((train, test, testclasses))
    return folddata


def savefftdatatofile(fftdata):
    """
    This function saves the data into a txt file.

    Parameters:
    -----------
    fftdata - fft data matrix (600,1000) which is generated by the function getfftdata()
    """
    np.savetxt('fftdata.txt', fftdata, '%f')


def loadfftdata(f):
    """
    This function loads the fft data from fft data file.

    Parameters:
    -----------
    f - fft data filename to be loaded
    """
    return np.loadtxt(f, float, '%f')


def savemfccdatatofile(mfccdata):
    """
    This function saves the data into a txt file.

    Parameters:
    -----------
    mfccdata - mfcc data matrix (600,13) which is generated by the function getmfccdata()
    """
    np.savetxt('mfccdata.txt', mfccdata, '%f')


def loadmfccdata(f):
    """
    This function loads the mfcc data from fft data file.

    Parameters:
    -----------
    f - mfcc data filename to be loaded
    """
    return np.loadtxt(f, float, '%f')


def normalize(data):
    """
    This function normalizes the data (both train, test).

    Parameters:
    -----------
    data - train or test data

    Returns:
    ---------
    data - Normalized train or test data

    Notes:
    ------
    Calculates the maximum value in a particular feature and divides all the values in that feature by the maximum value.
    Also puts value 1 before all the features to suffice w0. (Notes from Mitchell's document: To accommodate weight w0, we assume an illusory X0 = 1 for all l.)
    """
    for i in range(len(data[0])):
        maxv = 0
        for j in range(len(data)):
            if data[j][i] > maxv:
                maxv = data[j][i]
        for j in range(len(data)):
            data[j][i] = data[j][i]/maxv

    # To accommodate weight w0, we assume an illusory X0 = 1 for all l.
    for i in range(len(data)):
        data[i] = np.concatenate([[1], data[i]])
    return data


def trainfn(train, tempweights, eta, lmda):
    """
    This function generates the updated weight matrix based on the train data using
    single step of the gradient descent of the logistic regression algorithm

    Parameters:
    -----------
    train       - train or test data
    tempweights - weight matrix
    eta         - learning rate
    lmda        - constant that determines the strength of the penalty term

    Returns:
    ---------
    updatedweightmatrix - Updated weight matrix
    """
    deltamatrix = np.zeros((len(genres), len(train)))
    count = 0
    for i in range(len(genres)):
        for j in range(len(train)):
            if j >= count and j < count + 90:
                deltamatrix[i][j] = 1
        count += 90
    p = np.exp(tempweights.dot(np.transpose(train)))        # exp(W.transpose(X)) - gives Conditional data likelihood - P(Y/W, X)

    for i in range(len(p[0])):     # This loop sets 1's in the last row of matrix 'p'.
        p[len(p)-1][i] = 1

    for i in range(len(p[0])):     # Divide every column value by it's sum.
        p[:, i] = p[:, i]/np.sum(p[:, i])

    errormatrix = (deltamatrix - p).dot(train)                            # Calculate error matrix (Delta - P(Y/W, X)X)
    intermatrix = eta * (errormatrix - lmda * tempweights)                # eta * ((errormatrix - Wt)
    updatedweightmatrix = np.add(tempweights, intermatrix)                # Wt + intermatrix
    return updatedweightmatrix


def testfn(tempweights, test):
    """
    This function classifies the test data based on logistic regression algorithm.

    Parameters:
    -----------
    tempweights - trained weight matrix used to classify the songs.
    test        - test data from a fold that has to be classified.

    Returns:
    ---------
    newtestclasses - returns the classified labels of the test data.
    """
    p = np.exp(tempweights.dot(np.transpose(test)))         # exp(W.transpose(X)) - gives Conditional data likelihood - P(Y/W, X)

    for i in range(len(p[0])):                  # This loop sets 1's in the last row of matrix 'p'.
        p[len(p)-1][i] = 1

    for i in range(len(p[0])):                  # Divide every column value by it's sum.
        p[:, i] = p[:, i]/np.sum(p[:, i])

    newtestclasses = np.zeros((len(test), 1))   # matrix to store the classified labels of the genres.

    for i in range(len(p[0])):                  # This loop finds the maximum value in a column of P (i.e., column represents index of data) and corresponding maximum value index (row) is taken as the label (genre) of that data.
        maxv = 0
        for j in range(len(p)):
            if p[j][i] > maxv:
                maxv = p[j][i]
                maxj = j
        newtestclasses[i] = maxj                # Array to store the labels
    return newtestclasses


def calc_conf_acc(testclasses, newtestclasses):
    """
    This function computes the Confusion Matrix and Accuracy rate.

    Parameters:
    -----------
    testclasses     - test labels/classes from the genre data.
    newtestclasses  - computed test labels/classes from logistic regression algorithm.

    Returns:
    ---------
    confusionmatrix - Confusion matrix based on testclasses and newtestclasses.
    accuracy        - Accuracy rate
    """
    confusionmatrix = np.zeros((len(genres), len(genres)))
    correct = 0
    for i in range(len(testclasses)):
        o = int(testclasses[i])
        n = int(newtestclasses[i])
        confusionmatrix[o][n] += 1
        if testclasses[i] == newtestclasses[i]:
            correct += 1
    return confusionmatrix, float(correct)/len(testclasses)


def processdata(data, no_of_features):
    """
    This function is the entry point to start the computations.

    Parameters:
    -----------
    data            - fft/fft20/mfcc data
    no_of_features  - number of features that have to be computed.
    """
    folddata = kfold(data, 10)   # 10-fold cross validation
    eta = 0.01                      # Initializing learning rate
    eta_new = 0.01
    lmda = 0.001
    it = 300                        # Number of iterations for each fold to determine weight matrix
    eachfoldmaxaccuracies = []      # Array to store maximum accuracies obtained for each fold
    eachfoldmaxconfmatrices = []    # Array to store Confusion Matrix at maximum accuracies obtained for each fold
    for i in range(len(folddata)):              # Iterate over 10 folds of data
        weights = np.zeros((len(genres), no_of_features + 1))   # Initialize weights matrix with all zeros.
        train, test, testclasses = folddata[i]                  # Generate the k-fold data (10)
        train = normalize(train)                                # Normalize the train data
        test = normalize(test)                                  # Normalize the test data
        tempweights = weights[:]                                # Re-initialize weights matrix to all zeros.
        maxaccuracy = 0                                         # variable to store max-accuracy per fold.
        for j in range(it):                                     # Iterate the process for gradient descent (used in trainfn() function)
            print "Current Fold : " + str(i)
            print "Iteration : " + str(j)
            eta = eta_new / (1 + float(j) / it)                     # Calculate eta based on number of iterations
            tempweights = trainfn(train, tempweights, eta, lmda)    # generates the updated weight matrix based on the train data using single step of the gradient descent of the logistic regression algorithm
            newtestclasses = testfn(tempweights, test)              # classifies the test data based on the weight matrix obtained from the previous step
            confmatrix, accuracy = calc_conf_acc(testclasses, newtestclasses)   # Compute Confusion matrix and Accuracy
            if accuracy > maxaccuracy:                              # Calculate Maxaccuracy in the current fold and store the respective Confusion matrix in maxconfmatrix variable.
                maxaccuracy = accuracy
                maxconfmatrix = confmatrix
            print "Accuracy  : " + str(accuracy)
            print "Confusion Matrix : \n" + str(confmatrix)
        eachfoldmaxaccuracies.append(maxaccuracy)
        eachfoldmaxconfmatrices.append(maxconfmatrix)
    print "==============================================="
    for i in range(len(eachfoldmaxaccuracies)):                     # Print the max accuracy and respective confusion matrix for each fold.
        print "\n"
        print "Fold " + str(i) + " max accuracy : " + str(eachfoldmaxaccuracies[i])
        print "Confusion Matrix : "
        print eachfoldmaxconfmatrices[i]
    print "Avg of all folds accuracies : " + str(np.average(eachfoldmaxaccuracies))


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        print "Not enough arguments. Usage: -fft/-fft20/-mfcc <path to data>"
        exit(0)
    else:
        if os.path.isdir(args[1]):
            if args[0] == "-fft" or args[0] == "-fft20" or args[0] == "-mfcc":
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
                if args[0] == "-fft20":
                    print "Using fft20... 20 songs per each genre!"
                    print "Checking if fft data file (fftdata.txt) is already present..."
                    if os.path.isfile('fftdata.txt'):
                        print "fftdata.txt is already present. Using this file."
                        processdata(genbestfftdata(loadfftdata("fftdata.txt"), no_of_best_fft_features), no_of_best_fft_features)
                        exit(0)
                    else:
                        print "fftdata.txt is not present. Creating one."
                        savefftdatatofile(getfftdata(args[1]))
                        processdata(genbestfftdata(loadfftdata("fftdata.txt"), no_of_best_fft_features), no_of_best_fft_features)
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