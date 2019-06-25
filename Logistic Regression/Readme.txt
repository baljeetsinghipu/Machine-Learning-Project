# Project Title

Naive Bayes and Logisitic Regression implementation for spam and ham classification.

## Getting Started

You can find 4 files in the package:-

1. Readme file -- Which tell you how to run the files

2. Naive_Bayes.py -- Naive Bayes Implementation

3. logisitic_regression_with_stopping_words.py -- Logisitic regression before removing stopping words

4. logisitic_regression_without_stopping_words.py -- Logisitic regression after removing stopping words

### Following library needed:

import numpy as np : To ease calculations
import glob : for navigating the files
import os 
import sys
import string as st : Collection of string function
import pandas as pd : help to open test and train file
import csv: To avoid quotes error while opening file
from scipy import special : to avoid overflow while calculating e(-x) for very big number
import time : to calcualte execution time

You need to install the following library : 
1. pip install scipy
2. pip install numpy
3. pip install pandas


#### For the Naive_Bayes.py 

You have to pass two parameters only:-

1. Parameter 1st :- Training path
2. Parameter 2nd :- Test path

for example:- 

1. F:\Machine Learning\Assignment\Code\Assignment-2>Naive_bayes.py train test   // if the train, test files is in the same folder of python file

or 

2. F:\Machine Learning\Assignment\Code\Assignment-2>Naive_bayes.py "F:\Machine Learning\Assignment\Code\Assignment-2\train" "F:\Machine Learning\Assignment\Code\Assignment-2\test"

##### For the logisitic_regression_with_stopping_words.py or logisitic_regression_without_stopping_words.py

You have to pass 5 parameters:-

1. Parameter 1st: Training path
2. Parameter 2nd: Testing path
3. Parameter 3rd: lamda valule <Suggested: 2>
4. Parameter 4th: Learning Rate <Suggested: 0.001>
5. Parameter 5th: Iteration <Suggested: 200>

Bias is set to 0 by default.

for example:- 

F:\Machine Learning\Assignment\Code\Assignment-2>logisitic_regression_with_stopping_words.py "F:\Machine Learning\Assignment\Code\Assignment-2\train" "F:\Machine Learning\Assignment\Code\Assignment-2\test" 2 0.001 100


####Authors

Baljeet Singh