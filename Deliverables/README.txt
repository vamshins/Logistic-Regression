===========================================================================
CS 529 - Intro to Machine Learning - Assignment 3 - Logistic Regression
===========================================================================

1. Source code of the project is hosted on github:
	https://github.com/vamshins/Logistic-Regression

2. The executable "LogisticRegression.py" is provided in the assignment submission in UNM Learn.

3. I have programmed in "Python 2.7.9 |Anaconda 2.2.0 (64-bit)|" installed in Windows 8.1.
   Download all the files attached in UNM Learn submission and store them in "<HOME_DIR>" (This may be any folder on your OS)
   Copy the data into the "<HOME_DIR>/opihi.cs.uvic.ca" folder.
   The folder structure looks like -
			- <HOME_DIR>
					|--> /LogisticRegression.py
					|--> /fftdata.txt
					|--> /classesmatrixfft.txt
					|--> /mfccdata.txt
					|--> /classesmatrixmfcc.txt
					|--> /opihi.cs.uvic.ca/sound/genres/
						   |--> /classical
						   |--> /jazz
						   |--> /country
						   |--> /pop
						   |--> /rock
						   |--> /metal

4. Execution of the Program:
	4.1. Go to Run(Windows) or Terminal(Linux)
	4.2. Navigate to <HOME_DIR>
	4.3. Run the command -> python LogisticRegression.py <option> <path to data>
			 <option> 	- Give -fft or -fft20 or -mfcc
							-fft	-> Generates fft for the data and performs the Logistic Regression with Gradient Ascent
							-fft20	-> Generates 20 best features per genre for the fft data and performs the Logistic Regression with Gradient Ascent
							-mfcc	-> Generates mfcc for the data and performs the Logistic Regression with Gradient Ascent
		 <path to data> - Path to the music data. Eg: <HOME_DIR>\opihi.cs.uvic.ca\sound\genres

5. Output of the Program for each fold with highest accuracy:
	==========================================================================================
	For FFT:
	Execution : python LogisticRegression.py -fft <HOME_DIR>\opihi.cs.uvic.ca\sound\genres
	------------------------------------------------------------------------------------------		
		Fold 0 max accuracy : 0.533333333333
		Confusion Matrix : 
		[[ 10.   0.   0.   0.   0.   0.]
		 [  1.   3.   1.   0.   3.   2.]
		 [  3.   0.   7.   0.   0.   0.]
		 [  4.   0.   0.   0.   5.   1.]
		 [  2.   0.   0.   0.   7.   1.]
		 [  1.   1.   0.   0.   3.   5.]]


		Fold 1 max accuracy : 0.583333333333
		Confusion Matrix : 
		[[ 7.  0.  1.  1.  0.  1.]
		 [ 0.  2.  2.  3.  0.  3.]
		 [ 1.  1.  6.  1.  0.  1.]
		 [ 0.  0.  0.  9.  1.  0.]
		 [ 0.  1.  0.  1.  6.  2.]
		 [ 0.  3.  1.  1.  0.  5.]]


		Fold 2 max accuracy : 0.5
		Confusion Matrix : 
		[[ 7.  0.  2.  1.  0.  0.]
		 [ 0.  2.  2.  2.  1.  3.]
		 [ 2.  0.  6.  0.  0.  2.]
		 [ 1.  0.  0.  4.  0.  5.]
		 [ 0.  0.  1.  1.  4.  4.]
		 [ 1.  0.  0.  1.  1.  7.]]


		Fold 3 max accuracy : 0.55
		Confusion Matrix : 
		[[ 9.  0.  0.  0.  1.  0.]
		 [ 2.  3.  0.  1.  3.  1.]
		 [ 1.  1.  4.  1.  2.  1.]
		 [ 0.  0.  0.  4.  5.  1.]
		 [ 1.  0.  0.  0.  8.  1.]
		 [ 0.  0.  0.  1.  4.  5.]]


		Fold 4 max accuracy : 0.483333333333
		Confusion Matrix : 
		[[ 8.  0.  0.  0.  2.  0.]
		 [ 0.  1.  0.  0.  5.  4.]
		 [ 2.  0.  4.  0.  2.  2.]
		 [ 0.  0.  0.  1.  3.  6.]
		 [ 0.  0.  0.  2.  7.  1.]
		 [ 1.  0.  0.  0.  1.  8.]]


		Fold 5 max accuracy : 0.566666666667
		Confusion Matrix : 
		[[  7.   0.   1.   0.   2.   0.]
		 [  0.   2.   1.   0.   5.   2.]
		 [  0.   0.   7.   0.   3.   0.]
		 [  0.   0.   1.   4.   5.   0.]
		 [  0.   0.   0.   0.  10.   0.]
		 [  0.   1.   0.   2.   3.   4.]]


		Fold 6 max accuracy : 0.55
		Confusion Matrix : 
		[[  9.   0.   0.   1.   0.   0.]
		 [  1.   3.   0.   1.   3.   2.]
		 [  1.   3.   4.   0.   2.   0.]
		 [  0.   1.   1.   2.   5.   1.]
		 [  0.   0.   0.   0.  10.   0.]
		 [  0.   1.   1.   0.   3.   5.]]


		Fold 7 max accuracy : 0.533333333333
		Confusion Matrix : 
		[[ 7.  0.  1.  0.  2.  0.]
		 [ 0.  5.  0.  0.  4.  1.]
		 [ 1.  0.  5.  1.  2.  1.]
		 [ 0.  0.  0.  4.  5.  1.]
		 [ 0.  1.  0.  0.  8.  1.]
		 [ 0.  0.  1.  2.  4.  3.]]


		Fold 8 max accuracy : 0.55
		Confusion Matrix : 
		[[ 4.  1.  0.  0.  3.  2.]
		 [ 0.  8.  0.  0.  1.  1.]
		 [ 0.  1.  5.  0.  0.  4.]
		 [ 0.  0.  0.  3.  5.  2.]
		 [ 0.  1.  0.  0.  9.  0.]
		 [ 0.  1.  0.  2.  3.  4.]]


		Fold 9 max accuracy : 0.566666666667
		Confusion Matrix : 
		[[ 9.  0.  0.  1.  0.  0.]
		 [ 0.  2.  1.  0.  1.  6.]
		 [ 3.  1.  5.  0.  0.  1.]
		 [ 0.  0.  2.  4.  2.  2.]
		 [ 0.  1.  1.  2.  6.  0.]
		 [ 0.  1.  1.  0.  0.  8.]]
		Avg of all folds accuracies : 0.541666666667
		
	==========================================================================================
	For MFCC:
	Execution : python LogisticRegression.py -mfcc <HOME_DIR>\opihi.cs.uvic.ca\sound\genres
	------------------------------------------------------------------------------------------
	
		Fold 0 max accuracy : 0.733333333333
		Confusion Matrix : 
		[[ 10.   0.   0.   0.   0.   0.]
		 [  0.   6.   0.   1.   1.   2.]
		 [  3.   0.   6.   1.   0.   0.]
		 [  0.   1.   0.   9.   0.   0.]
		 [  0.   1.   0.   0.   8.   1.]
		 [  0.   0.   1.   4.   0.   5.]]


		Fold 1 max accuracy : 0.65
		Confusion Matrix : 
		[[ 8.  0.  0.  1.  0.  1.]
		 [ 0.  6.  2.  0.  2.  0.]
		 [ 2.  3.  2.  2.  1.  0.]
		 [ 0.  0.  0.  9.  0.  1.]
		 [ 1.  1.  0.  0.  8.  0.]
		 [ 0.  4.  0.  0.  0.  6.]]


		Fold 2 max accuracy : 0.683333333333
		Confusion Matrix : 
		[[ 8.  0.  0.  0.  0.  2.]
		 [ 0.  4.  3.  1.  1.  1.]
		 [ 2.  0.  6.  0.  1.  1.]
		 [ 0.  0.  0.  9.  0.  1.]
		 [ 0.  1.  0.  0.  9.  0.]
		 [ 0.  1.  0.  3.  1.  5.]]


		Fold 3 max accuracy : 0.683333333333
		Confusion Matrix : 
		[[  8.   1.   1.   0.   0.   0.]
		 [  0.   6.   0.   0.   1.   3.]
		 [  1.   3.   4.   0.   0.   2.]
		 [  0.   0.   0.  10.   0.   0.]
		 [  0.   2.   0.   0.   7.   1.]
		 [  0.   2.   0.   0.   2.   6.]]


		Fold 4 max accuracy : 0.7
		Confusion Matrix : 
		[[  8.   1.   1.   0.   0.   0.]
		 [  2.   4.   1.   0.   2.   1.]
		 [  1.   3.   5.   0.   0.   1.]
		 [  0.   0.   0.  10.   0.   0.]
		 [  0.   0.   0.   0.  10.   0.]
		 [  0.   0.   0.   4.   1.   5.]]


		Fold 5 max accuracy : 0.7
		Confusion Matrix : 
		[[  9.   0.   1.   0.   0.   0.]
		 [  0.   6.   2.   0.   1.   1.]
		 [  3.   1.   5.   0.   0.   1.]
		 [  0.   0.   0.   9.   0.   1.]
		 [  0.   0.   0.   0.  10.   0.]
		 [  1.   1.   3.   2.   0.   3.]]


		Fold 6 max accuracy : 0.716666666667
		Confusion Matrix : 
		[[  8.   1.   1.   0.   0.   0.]
		 [  2.   5.   3.   0.   0.   0.]
		 [  3.   0.   4.   0.   2.   1.]
		 [  0.   0.   1.   9.   0.   0.]
		 [  0.   0.   0.   0.  10.   0.]
		 [  0.   0.   1.   2.   0.   7.]]


		Fold 7 max accuracy : 0.783333333333
		Confusion Matrix : 
		[[  8.   0.   2.   0.   0.   0.]
		 [  1.   7.   1.   0.   0.   1.]
		 [  0.   1.   8.   0.   1.   0.]
		 [  0.   0.   1.   9.   0.   0.]
		 [  0.   0.   0.   0.  10.   0.]
		 [  0.   1.   0.   4.   0.   5.]]


		Fold 8 max accuracy : 0.766666666667
		Confusion Matrix : 
		[[ 10.   0.   0.   0.   0.   0.]
		 [  2.   6.   1.   0.   0.   1.]
		 [  2.   1.   6.   0.   1.   0.]
		 [  0.   0.   0.  10.   0.   0.]
		 [  0.   1.   0.   0.   9.   0.]
		 [  0.   1.   0.   3.   1.   5.]]


		Fold 9 max accuracy : 0.75
		Confusion Matrix : 
		[[  8.   0.   1.   0.   0.   1.]
		 [  1.   7.   0.   1.   1.   0.]
		 [  2.   1.   6.   0.   1.   0.]
		 [  0.   0.   0.  10.   0.   0.]
		 [  0.   1.   0.   0.   8.   1.]
		 [  0.   0.   2.   0.   2.   6.]]
		 
		Avg of all folds accuracies : 0.716666666667
