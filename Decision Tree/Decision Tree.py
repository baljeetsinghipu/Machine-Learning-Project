# -*- coding: utf-8 -*-

# Decision Tree Implementation using  Information gain heuristic and Variance impurity heuristic 

#Author ---Baljeet Singh

from __future__ import print_function
import copy
import random
import sys
import os
counter=0
cutcounter = 0
counter_one=0
counter_zero=0
flag=0
training_data=[]
data=[]
class print_command_error(Exception):
    pass

try:
    L=int(sys.argv[1])
    K=int(sys.argv[2])
    training_set=sys.argv[3]
    validation_set =sys.argv[4]  
    test_set = sys.argv[5]
    print_command = sys.argv[6].lower()
    try:
        if (print_command=="yes" or print_command=="no"):
            pass
        else:
            raise print_command_error
            
    except print_command_error:
        print("Only possible value for 'Print the Decision tree' is either 'Yes' or 'No' <case insensitive>")
        os._exit(0)
        
except:
    print("Error: You have not provided all the 6 arguments")
    print("Please provide in following order")
    print("<file_name.py> <L> <K> <training_data_path> <validation_data_path> <test_data_path> <print: Yes or No>")
    os._exit(0)
    

#Function to get the data from Validation, Test and Training File
def fileopener(filename):
    global data
    data=[]
    file_data=[]
    file1 = open(filename, "r")
    for line in file1:
        line = line.lstrip()
        data.append(line.split(','))
    
    for i in range (len(data)):
        data[i][-1] = data[i][-1].strip()
    
    for i in range(len(data)):
        file_data.append(data[i])
    file_data.pop(0)
    
    return(file_data)


training_data= fileopener(training_set)
header = data[0]
test_data=fileopener(test_set)
validation_data = fileopener(validation_set)


gain_list=[i for i in range(0,(len(training_data[0])-1))]

# Function to calcuolate how many 1 and O occurs in Data
def attribute_count(rows):
    count = {}
    for row in rows:
        label=row[-1];
        if label not in count:
            count[label] = 0
        count[label] = count[label]+1;
    return count


def __init__(self, attribute, value):
    self.attribute = attribute;

#function to split row based on input column number    
def split(rows,col):
    true_list=[]
    false_list=[]
    
    for row in rows:
#        print(col)
        if row[col] == "1":
            true_list.append(row)
        else:
            false_list.append(row)
    return true_list, false_list


counter1 =0    
#function to choose the best attribute
def find_best_split(rows,ent,IG):
    
    best_gain = 0  # keep track of the best information gain
    best_question = None
    best_col = None # keep train of the feature / value that produced it
    n_features = len(rows[0]) -1  # number of columns
    
    for col in range(n_features):
        if col not in gain_list:
            continue
        question = Question(col, header[col])
        true_rows=[]
        false_rows=[]
        true_rows,false_rows = split(rows,col)
        
        if len(true_rows)==0 or len(false_rows)==0:
            continue
        if IG==1:
            attribute = calculate_entropy(true_rows,false_rows)
        elif IG==0:
            attribute = calculate_variance(true_rows,false_rows)
            
        gain=float(ent)-float(attribute)
        if gain >= best_gain:
            best_gain, best_question,best_col= gain, question,col       
    return best_gain, best_question,best_col

#function to intify whether the variable is numeric or not
def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

#class question to save value and column number of non-leaf node
class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, row):
        val = row[self.column]
        return val == self.value 
    def match1(self, example):
        val = example[self.column]
        if val==0:
            return 0
        else:
            return 1

#function to caluculate entropy in case of Information gain
def calculate_entropy(true_rows,false_rows):
    postive_list1=[]
    negative_list1=[]
    postive_list2=[]
    negative_list2=[]
    entropy=0
    
    for row in true_rows:
        if(row[-1]=="1"):
            postive_list1.append(row)
        else:
            negative_list1.append(row)
    
    for row in false_rows:
        if(row[-1]=="1"):
            postive_list2.append(row)
        else:
            negative_list2.append(row)
    
    total = len(true_rows)+len(false_rows)
    entropy= (len(true_rows)/total* entropy_finder(len(postive_list1),len(negative_list1))) + ((len(false_rows)/total)* entropy_finder(len(postive_list2),len(negative_list2)))
    
    return entropy

#Funstion to calcualte variance in case of variance
def calculate_variance(true_rows,false_rows):
    postive_list1=[]
    negative_list1=[]
    postive_list2=[]
    negative_list2=[]
    variance=0
    
    for row in true_rows:
        if(row[-1]=="1"):
            postive_list1.append(row)
        else:
            negative_list1.append(row)
    
    for row in false_rows:
        if(row[-1]=="1"):
            postive_list2.append(row)
        else:
            negative_list2.append(row)
    
    total = len(true_rows)+len(false_rows)
    variance= (len(true_rows)/total* variance_finder(len(postive_list1),len(negative_list1))) + ((len(false_rows)/total)* variance_finder(len(postive_list2),len(negative_list2)))
    
    return variance


#fundtion to calculate entropy of class
def class_entropy(true_rows,false_rows):
    from math import log
    log2 = lambda x:log(x)/log(2)
    total= len(true_rows) + len(false_rows)
    if(len(true_rows) ==0 or len(false_rows)==0):
        entropy=0
    else:
        entropy=- (len(true_rows)/total)*log2(len(true_rows)/total) - (len(false_rows)/total)*log2(len(false_rows)/total)
    return entropy

#function to calculate variance of class
def class_variance(true_rows,false_rows):
    total= len(true_rows) + len(false_rows)
    
    if(len(true_rows) ==0 or len(false_rows)==0):
        variance=0
    else:
        variance = (len(true_rows) * len(false_rows))/ (total*total)
    return variance

#Function to find variance of each attributes
def variance_finder(positive,negative):
    
    total= positive + negative
    if(positive ==0 or negative==0):
        variance=0
    else:
        variance= positive*negative/(total*total)
        
    return variance

   #Function to find entropy of each attributes 
def entropy_finder(positive,negative):
#    print(positive,negative)
    from math import log

    log2 = lambda x:log(x)/log(2)
    if(positive ==0 or negative==0):
        entropy=0
    else:
        entropy=-((positive/(positive+negative))*log2(positive/(positive+negative)))-((negative/(positive+negative))*log2(negative/(positive+negative)))
    return entropy

def value_extract(rows):
#    final_list=[]
    s=""
    result=attribute_count(rows)
    for r in result:
        if s=="":
            s+=str(r)
    return s

#class for leaf node  
class Leaf:
    def __init__(self, rows):
        if (is_numeric(rows)):
            self.value = rows
        else:
            self.value= value_extract(rows)
         

counter=1

#class for non-leaf node
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        
        
test_list=[]
false_flag=1    

#function to build tree using variance or Information gain
def build_tree(rows,IG):
    global counter
    flad=0
    if flad==0:
        counter=0
        flad=1
    global false_flag
    global gain_list
    true,false = split(rows,-1)
    if IG==1:
        ent = class_entropy(true,false)
        if ent == 0:
            return Leaf(rows)
        gain,question,col= find_best_split(rows,ent,IG);
    elif IG==0:
        var = class_variance(true,false)
        if var == 0:
            return Leaf(rows)
        gain,question,col= find_best_split(rows,var,IG);
        
    test_list.append(col)
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = [], []
    true_rows, false_rows = split(rows, col);

    true_branch = build_tree(true_rows,IG)
    if(false_flag==1):
        gain_list=[i for i in range(0,(len(training_data[0])-1))]
    false_flag=0
    false_branch = build_tree(false_rows,IG)
    return Decision_Node(question,true_branch,false_branch)

#function to get the column or header details of given data set
def get_column(value):
    for i in range(0, len(header)-1):
        if header[i]== value:
            return i
#function to match 0 or 1 in the given test file against training set
def trace(row,node):

    if isinstance(node, Leaf):
        return (node.value)
    
    col= get_column(node.question.value)
    if row[col]=="1":
        return trace(row, node.true_branch)
    else:
        return trace(row, node.false_branch)
    
# Function to Print Tree in suitable formatting   
def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return

    if isinstance(node.true_branch, Leaf):
        print (spacing + str(node.question.value) + " = 1 : "+ str(node.true_branch.value)  )
    else:
        print (spacing + str(node.question.value) + " = 1 : " )
    print_tree(node.true_branch, spacing + "|  ")

    if isinstance(node.false_branch, Leaf) :
        print (spacing + str(node.question.value) + " = 0 : "+ str(node.false_branch.value))
    else:
        print (spacing + str(node.question.value) + " = 0 : " )
    print_tree(node.false_branch, spacing + "|  ")


#Function to update tree after pruning
def update(tree,P,value):
    global cutcounter
    if isinstance(tree, Leaf):
        return Leaf(tree.value)
    else:
        cutcounter=cutcounter+1
        
        if(cutcounter == P):
            return Leaf(value)
        else:
            true=update(tree.true_branch,P,value)
            false=update(tree.false_branch,P,value)
    return Decision_Node(tree.question,true,false)

# Function to find the accurency of tree with respect to give data set
def accuracy(tree,data):
    count=0;
    for test_data in data:
#        print(test_data)
        result = trace(test_data, tree)
#        print (result)
        if(result==test_data[-1]):
            count=count+1
    return(str(count/(len(data))*100))


#Function to calculate non leaf members in the given tree       
def add_nonleaf_number(new_tree):
    global non_leaf_counter 
    if isinstance(new_tree, Leaf):
        return
    else:
        non_leaf_counter =non_leaf_counter+1
        add_nonleaf_number(new_tree.true_branch)
        add_nonleaf_number(new_tree.false_branch)
    return non_leaf_counter

#Function to find the majority of class and replace it with non leaf node in case of pruning    
def majority(node, P):
    global counter_one
    global counter_zero
    global flag
    global counter
    if isinstance(node, Leaf):
        if (node.value=="1" and flag==1):
            counter_one = counter_one +1
        elif(node.value=="0" and flag==1):
            counter_zero=counter_zero+1
        return  
    else:
        counter=counter+1
        if(counter==P):
            flag=1
            majority(node.true_branch,P)
            majority(node.false_branch,P)
            flag=0           
        else:
            majority(node.true_branch,P)
            majority(node.false_branch,P)
    if counter_one>counter_zero:
        result=1
    elif counter_zero>counter_one:
        result=0
    else:
        result=1
    return result

#Function to prun the tree
def post_pruning(L, K, node,validation_data):
    global cutcounter
    global non_leaf_counter
    global counter_one
    global counter_zero
    global flag
    global counter
    new_accuracy=0
    old_accuracy = round(float(accuracy(node,validation_data)),2)
    best_tree = node
    for i in range(1,L):
        new_tree = copy.deepcopy(node)
        M = random.randint(1, K)
        for j in range(1,M):
            non_leaf_counter=0
            stop=int(add_nonleaf_number(new_tree)) - 1
            if stop<2:
                continue
            P = random.randint(2, stop)
            counter_one=0
            counter_zero=0
            flag=0
            counter=0
            value= majority(new_tree,P)
            value=int(value)
            cutcounter=0
            new_tree=update(new_tree,P,value)        
            new_accuracy=accuracy(new_tree,validation_data)            
            new_accuracy=round(float(new_accuracy),2)        
            if(new_accuracy>old_accuracy):
                best_tree=new_tree

    return best_tree

#Function which will call the pruning method after calling the build tree fundtion
def start(L,K,validation_data,test_data,print_command):
    my_tree_IG = build_tree(training_data,1)
    old_accuracy = accuracy(my_tree_IG,test_data)
    old_accuracy=round(float(old_accuracy),2)
    print("")
    print("--------------------------------------------------------------------")
    print("|                         Before Pruning                           |")
    print("--------------------------------------------------------------------")
    print("")
    print("Accuracy of Decision tree with 'Information gain  heuristic': " + str(old_accuracy))
    my_tree_variance = build_tree(training_data,0)
    old_accuracy = accuracy(my_tree_variance,test_data)
    old_accuracy=round(float(old_accuracy),2)
    print("Accuracy of Decision tree with 'Variance impurity heuristic': " + str(old_accuracy))
    print("")
    print("--------------------------------------------------------------------")
    print("|                         After Pruning                            |")
    print("--------------------------------------------------------------------")
    print("")
    best_tree_IG=post_pruning(L,K,my_tree_IG, validation_data)
    new_accuracy = accuracy(best_tree_IG,test_data)
    new_accuracy=round(float(new_accuracy),2)
    print("Accuracy of Decision tree with 'Information gain  heuristic': " + str(new_accuracy))
    best_tree_Variance=post_pruning(L,K,my_tree_variance, validation_data)
    new_accuracy = accuracy(best_tree_Variance,test_data)
    new_accuracy=round(float(new_accuracy),2)
    print("Accuracy of Decision tree with 'Variance impurity heuristic': " + str(new_accuracy))
    
    if(print_command=="yes"):
        print("------------------------------------------------------------------")
        print("Best tree after Pruning process (using Information gain heuristic)|")
        print("------------------------------------------------------------------")
        print_tree(best_tree_IG)
        print("--------------------------------------------------------------------")
        print("Best tree after Pruning process (using Variance impurity heuristic )|")
        print("--------------------------------------------------------------------")
        print_tree(best_tree_Variance)
        

#calling main function
start(L,K,validation_data,test_data,(print_command.lower()))