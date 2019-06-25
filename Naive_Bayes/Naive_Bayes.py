from __future__ import division
import math as mt
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from collections import Counter
import os
import csv
global spam,ham,denom,total_file_count,spam_file_count,ham_file_count
import sys
#print(string.punctuation)
train_link=sys.argv[1]
test_link=sys.argv[2]
train_spam_link= train_link+"\spam\*.txt"
train_ham_link=train_link+"\ham\*.txt"
test_spam_link=test_link+"\spam\*.txt"
test_ham_link=test_link+"\ham\*.txt"
spam_clean_list=[]
ham_clean_list=[]
spam_counter= {}
ham_counter= {}

#stop_words=['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
stop_words = []
import glob


def log_prior_probability():
    global total_file_count,spam_file_count,ham_file_count
    spam_file_count=len(glob.glob(train_spam_link))
    ham_file_count=len(glob.glob(train_ham_link))
    total_file_count= ham_file_count+ spam_file_count
    spam_prior_probability= mt.log(spam_file_count)-mt.log(total_file_count)
    ham_prior_probability = mt.log(ham_file_count)-mt.log(total_file_count)
    return spam_prior_probability, ham_prior_probability

#
def fileopener(filename):
#    print(filename)
    if(os.stat(filename).st_size == 0):
        return " "
    data = pd.read_csv(filename,delimiter='\t',sep='\t' ,error_bad_lines=False, squeeze=True,header=None,skip_blank_lines=True, encoding='Latin-1',engine='c', quoting=csv.QUOTE_NONE)
    result=data.to_string(index=False)
    data= result.split()
    return data


spam = [fileopener(filename) for filename in glob.glob(train_spam_link)]
ham = [fileopener(filename) for filename in glob.glob(train_ham_link)]
spam1 = [fileopener(filename) for filename in glob.glob(train_spam_link)]
ham1 = [fileopener(filename) for filename in glob.glob(train_ham_link)]
#print(spam)


def stop_words_checker(word):

    if(word not in stop_words):
        return 1
    else:
        return 0

    
def cleaner(file_object):
#    print(len(stop_words))
    temp_file=file_object
    temp_list=[]
    for i in range(0,len(temp_file)):
        for j in range(0, len(temp_file[i])):
            if (stop_words_checker(temp_file[i][j].lower())):
                temp_list.append((temp_file[i][j]).lower())
    return temp_list


def cleaner2(file_object):
    temp_file=file_object
    temp_list=[]
    for i in range(0,len(temp_file)):
        if (stop_words_checker(temp_file[i].lower())):
                temp_list.append((temp_file[i]).lower())
    return temp_list




def frequency_counter(list):
#    print(len(Counter(list)))
    return Counter(list)





def frequency_founder(file_name, word):
    
    for key,value in file_name.items():
        if key==word:
            return value
        
    return 0


def common_words_counter(spam_counter, ham_counter):
    temp_list=[]
    for key,value in spam_counter.items():
        temp_list.append(key)
    for key,value in ham_counter.items():
        if key not in temp_list:
            temp_list.append(key)
    return len(temp_list)
        
    
def probability_calculator(file, word, denom,clean_list):
    
    times= frequency_founder(file, word)

    log_probability= mt.log((times + 1 ))- mt.log( (len(clean_list) + denom))

    return log_probability
    

def prereq():
    global spam,ham,denom
    spam,ham=log_prior_probability()
    denom=common_words_counter(spam_counter,ham_counter)

def class_probability(file, text_list, spam_enable,clean_list):
#    global spam,ham,denom
    total = mt.log(1)
    for i in range(len(text_list)):
        probability=probability_calculator(file, text_list[i],denom,clean_list)
        total = total + probability
    
    if(spam_enable=="True"):
        total = total + spam
    elif(spam_enable=="False"):
        total = total + ham
    return total

def class_finder(spam_file, ham_file, text):
    text_list_temp = text.split()
    text_list = cleaner2(text_list_temp)
    spam_rate=class_probability(spam_file, text_list, "True",spam_clean_list)
    ham_rate= class_probability(ham_file, text_list, "False",ham_clean_list)
    if (spam_rate>ham_rate ):
        return "True"
    elif(spam_rate<ham_rate ):
        return "False"
    else:

        return "NA"


def text_extraction(filename):
    if(os.stat(filename).st_size == 0):
        return " "
    data = pd.read_csv(filename,delimiter='\t',sep='\t' ,error_bad_lines=False, squeeze=True,header=None,skip_blank_lines=True, encoding='Latin-1',engine='c', quoting=csv.QUOTE_NONE)
    result=data.to_string(index=False)
    return result
    

def accuracy(list_of_files,decide):
    spam_ac_counter=0
    file_ac_counter=0
    prereq()
    for i in range(len(list_of_files)):
#        print(list_of_files[i])
        text=text_extraction(list_of_files[i])
        text=text.lower()
        if(class_finder(spam_counter, ham_counter, text)==decide ):
            spam_ac_counter=spam_ac_counter+1
        file_ac_counter = file_ac_counter + 1
    accuracy = (spam_ac_counter/file_ac_counter)*100

    return accuracy

def add_stop_words():
    
    global stop_words
    stop_words=["a",
"about",
"above",
"after",
"again",
"against",
"all",
"am",
"an",
"and",
"any",
"are",
"aren't",
"as",
"at",
"be",
"because",
"been",
"before",
"being",
"below",
"between",
"both",
"but",
"by",
"can't",
"cannot",
"could",
"couldn't",
"did",
"didn't",
"do",
"does",
"doesn't",
"doing",
"don't",
"down",
"during",
"each",
"few",
"for",
"from",
"further",
"had",
"hadn't",
"has",
"hasn't",
"have",
"haven't",
"having",
"he",
"he'd",
"he'll",
"he's",
"her",
"here",
"here's",
"hers",
"herself",
"him",
"himself",
"his",
"how",
"how's",
"i",
"i'd",
"i'll",
"i'm",
"i've",
"if",
"in",
"into",
"is",
"isn't",
"it",
"it's",
"its",
"itself",
"let's",
"me",
"more",
"most",
"mustn't",
"my",
"myself",
"no",
"nor",
"not",
"of",
"off",
"on",
"once",
"only",
"or",
"other",
"ought",
"our",
"ours",
"ourselves",
"out",
"over",
"own",
"same",
"shan't",
"she",
"she'd",
"she'll",
"she's",
"should",
"shouldn't",
"so",
"some",
"such",
"than",
"that",
"that's",
"the",
"their",
"theirs",
"them",
"themselves",
"then",
"there",
"there's",
"these",
"they",
"they'd",
"they'll",
"they're",
"they've",
"this",
"those",
"through",
"to",
"too",
"under",
"until",
"up",
"very",
"was",
"wasn't",
"we",
"we'd",
"we'll",
"we're",
"we've",
"were",
"weren't",
"what",
"what's",
"when",
"when's",
"where",
"where's",
"which",
"while",
"who",
"who's",
"whom",
"why",
"why's",
"with",
"won't",
"would",
"wouldn't",
"you",
"you'd",
"you'll",
"you're",
"you've",
"your",
"yours",
"yourself",
"yourselves",
]


def main1(test_spam_link):
    global stop_words,spam_clean_list,spam_counter,ham_clean_list,ham_counter
    spam_clean_list=cleaner(spam)
    ham_clean_list=cleaner(ham)
    spam_counter= frequency_counter(spam_clean_list)
    ham_counter= frequency_counter(ham_clean_list)
    
    print("Before removing stop words")
    print("--------------------------")
    
    print("Spam Accuracy: "+ str(round(accuracy(glob.glob(test_spam_link),"True"),2))+" %")
    print("Ham Accuracy: "+ str(round(accuracy(glob.glob(test_ham_link),"False"),2))+" %")
    print("")
    
    add_stop_words()
    spam_clean_list=cleaner(spam1)
#    print(spam_clean_list)
    ham_clean_list=cleaner(ham1)
    spam_counter= frequency_counter(spam_clean_list)
    ham_counter= frequency_counter(ham_clean_list)
    
    print("After removing stop words")
    print("--------------------------")
    
    print("Spam Accuracy: "+ str(round(accuracy(glob.glob(test_spam_link),"True"),2))+" %")
    print("Ham Accuracy: "+ str(round(accuracy(glob.glob(test_ham_link),"False"),2))+" %")
    
    
    
    
    
    
main1(test_spam_link)