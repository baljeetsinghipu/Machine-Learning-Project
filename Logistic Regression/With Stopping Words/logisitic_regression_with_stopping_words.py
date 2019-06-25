# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 02:42:04 2018

@author: Baljeet Singh
"""
from scipy import special
import numpy as np
import glob 

import os
import sys
import string as st
import pandas as pd
import csv
np.set_printoptions(threshold=sys.maxsize)
from collections import Counter
import time
#import pandas as pd

lambda_parameter = int(sys.argv[3])
learningrate = float(sys.argv[4])
iteration=int(sys.argv[5])

#print(lambda_parameter, learningrate, iteration)
pd.set_option('display.max_colwidth', -1)

train_link=sys.argv[1]
test_link=sys.argv[2]
train_spam_link= train_link+ "\spam\*.txt"
train_ham_link=train_link+"\ham\*.txt"
test_spam_link=test_link+"\spam\*.txt"
test_ham_link=test_link+"\ham\*.txt"

def fileopener(filename):
    if(os.stat(filename).st_size == 0):
        return " "
    data = pd.read_csv(filename,delimiter='\t',sep='\t' ,error_bad_lines=False, squeeze=True,header=None,skip_blank_lines=True, encoding='Latin-1',engine='c', quoting=csv.QUOTE_NONE)
    result=data.to_string(index=False)
    data= result.split()
    return data


def cleaner1(file_object):
    temp_file=file_object
    temp_list=[]
    temp_list1=[]
        
    for i in range(0,len(temp_file)):
        for j in range(0, len(temp_file[i])):
            temp_list.append(temp_file[i][j].lower())
        temp_list1.append(temp_list)
        temp_list=[]
    return temp_list1

def cleaner2(file_object):
    temp_file=file_object
    temp_list=[]
    for i in range(0,len(temp_file)):
        for j in range(0, len(temp_file[i])):
            temp_list.append(temp_file[i][j].lower())   
    return temp_list

def frequency_counter(list):
    return Counter(list)

def common_words_counter(spam_clean_list, ham_clean_list):
    temp_list=[]
    for i in spam_clean_list:
        if i not in temp_list:
            temp_list.append(i)
    for i in ham_clean_list:
        if i not in temp_list:
            temp_list.append(i)
    return temp_list


def tokenization(word):
    if(word not in st.punctuation):
        return 1
    else:
        return 0

train_spam = [fileopener(filename) for filename in glob.glob(train_spam_link)]
train_ham = [fileopener(filename) for filename in glob.glob(train_ham_link)]
test_spam = [fileopener(filename) for filename in glob.glob(test_spam_link)]
test_ham = [fileopener(filename) for filename in glob.glob(test_ham_link)]
train_spam_filename = glob.glob(train_spam_link)
train_ham_filename = glob.glob(train_ham_link)
test_spam_filename = glob.glob(test_spam_link)
test_ham_filename = glob.glob(test_ham_link)
train_total_filename = train_spam_filename + train_ham_filename
test_total_filename = test_spam_filename + test_ham_filename
spam_file_count=int(len(glob.glob(train_spam_link)))
ham_file_count=int(len(glob.glob(train_ham_link)))
total_file_count= ham_file_count+ spam_file_count
test_spam_file_count=int(len(glob.glob(test_spam_link)))
test_ham_file_count=int(len(glob.glob(test_ham_link)))
test_total_file_count= test_ham_file_count+ test_spam_file_count
o_spam_clean_list=cleaner1(train_spam)
test_o_spam_clean_list=cleaner1(test_spam)
o_ham_clean_list= cleaner1(train_ham)
test_o_ham_clean_list=cleaner1(test_ham)
o_total_clean_list=o_spam_clean_list+o_ham_clean_list

#print(o_total_clean_list[2])
test_o_total_clean_list=test_o_spam_clean_list+test_o_ham_clean_list
spam_clean_list=cleaner2(o_spam_clean_list)
test_spam_clean_list=cleaner2(test_o_spam_clean_list)
ham_clean_list= cleaner2(o_ham_clean_list)
test_ham_clean_list= cleaner2(test_o_ham_clean_list)
common_list=common_words_counter(spam_clean_list,ham_clean_list)
test_common_list=common_words_counter(test_spam_clean_list,test_ham_clean_list)

features_matrix = np.zeros((total_file_count,len(common_list)))
test_features_matrix = np.zeros((test_total_file_count,len(test_common_list)))
weight_matrix= [0.0 for y in range(len(common_list))]

bias=0

spam_value=[1 for i in range(spam_file_count)]
ham_value=[0 for i in range(ham_file_count)]
test_spam_value=[1 for i in range(test_spam_file_count)]
test_ham_value=[0 for i in range(test_ham_file_count)]

target_matrix= spam_value+ham_value
test_target_matrix= test_spam_value+test_ham_value


def matrix_creation(total_clean_list, featuresmatrix, commonlist):
    for i in range(len(total_clean_list) ):
        for j in range(len(total_clean_list[i])):
            index=commonlist.index(total_clean_list[i][j])
            frequency = total_clean_list[i].count(total_clean_list[i][j])
            featuresmatrix[i][index]= frequency

matrix_creation(o_total_clean_list,features_matrix,common_list)
matrix_creation(test_o_total_clean_list,test_features_matrix,test_common_list)
sigsum=[0.0 for i in range(total_file_count)]

def sigmoid(x):
    try:
#        print("I run")
        
        
#        sig = 1/(1 + np.exp(-x))
        sig = special.expit(x)
#        print("I m also run")
        return sig
    except Exception:
        return 0

def sigmoidcall(file_count):
    global sigsum
    
    error=0 
    
    for i in range(file_count):
        sum1 = 1.0
        sigmoidsum=0.0
        
        for j in range(len(common_list)):
            sum1+= weight_matrix[j]* features_matrix[i][j]
        
#        print(sum1)
        sigmoidsum=sigmoid(sum1)
#        print(sigmoidsum)
        sigsum[i] = sigmoidsum
        
        error+=sigsum[i]
        
        


def weight_update(length_weight_matrix):
    global sigsum
    error=0
#    diff_in_error=0
    
    for i in range(length_weight_matrix):
        diffval=bias
        
        for fileno in range(total_file_count):
            word_count =features_matrix[fileno][i]
            targetvalue=target_matrix[fileno]
            sigsum1=sigsum[fileno]
            diffval+= word_count*(targetvalue-sigsum1)
        
        prevweight= weight_matrix[i]
        weight_matrix[i]+= ((diffval*learningrate) - (learningrate*lambda_parameter*prevweight))
    
    return error,diffval

def train():
    start_time = time.time()

    print("Training start")
    for i in range(iteration):
        print("Iteration - "+str(i+1))
        sigmoidcall(total_file_count)
        error,diffval=weight_update(len(weight_matrix))
    print("Training end")
    print("--- %s seconds ---" % (time.time() - start_time))
        

def classify():
    
    positive_hamcount=0
    negative_hamcount=0
    positive_spamcount=0
    negative_spamcount=0
    
    for fileno in range(len(test_total_filename)):
        sum = 1.0
        sigmoidsum = 0.0
        #for each word
        for i in range(len(test_common_list)):
            word= test_common_list[i]
            
            if word in common_list:
                index= common_list.index(word)
                weight= weight_matrix[index]
                wordcount= test_features_matrix[fileno][i]
                
                sum+= weight*wordcount
        
#        print(sum)
        sigmoidsum = sigmoid(sum)
        classify=0
        if(test_target_matrix[fileno]==0):
            if sigmoidsum<0.5:
                positive_hamcount+=1.0
            else:
                negative_hamcount+=1.0
                classify=1
        else:
            if sigmoidsum>=0.5:
                positive_spamcount +=1.0
                classify=1
            else:
                negative_spamcount+=1.0
    
    print("Accuracy ham file:"+str(round((positive_hamcount /(positive_hamcount+negative_hamcount))*100,3)))

    print("Accuracy spam file:"+str(round((positive_spamcount/(positive_spamcount+negative_spamcount))*100,3)))
    

def main():
    start_time = time.time()
    train()
    print("Classification Start")
    classify()
    print("--- %s seconds ---" % (time.time() - start_time))        

main()
            



