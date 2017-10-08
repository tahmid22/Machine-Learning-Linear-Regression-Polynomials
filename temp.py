# -*- coding: utf-8 -*-

import numpy
from numpy import matrix
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

#Question 1
###############################################################################

def multiply(matr_a, matr_b):
    '''
    :param matr_a: The matrix A represented as lists of list
    :param matr_b: The matrix B represented as lists of list
    :return: return_matr: matrix to return after done mulplying A and N
    '''
    if len(matr_a[0]) != len(matr_b):
        return "The matrices cannot be multiplied"

    final_rows = len(matr_a)
    final_cols = len(matr_b[0])
    return_matr = [[0 for x in range(final_cols)] for y in range(final_rows)]

    for k in range (0, final_rows):
        for i in range (0, final_cols):
            value = 0
            for j in range (0, len(matr_b)):
                value += matr_a[k][j]*matr_b[j][i]
            return_matr[k][i] = value

    return return_matr
'''
matrixA = matrix([[1,2],[3,4],[5,6]]) 
matrixB = matrix([[7,8,9],[10,11,12]])
print(multiply(matrixA, matrixB))
'''

def mymeasure(I, K, J):
    #I x K matrix:
    rows, cols = I, K
    matrix1 = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(numpy.random.rand())
        matrix1.append(col)
        
    #K x JK matrix:
    rows, cols = K, J
    matrix2 = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(numpy.random.rand())
        matrix2.append(col)
        
    #compute time & compute the difference:
    start_time = time.time()
    numpy.matmul(matrix1, matrix2)
    finish_time = time.time()
    time_took_matmul = finish_time - start_time
    
    start_time = time.time()
    multiply(matrix1, matrix2)
    finish_time = time.time()
    time_took_mymult = finish_time - start_time
    
    diff = time_took_matmul - time_took_mymult
    print (diff)



#Question 3
###############################################################################     

def dataMatrix(X, M):
    return_matr = []
    for i in range (0, len(X)):
        return_matr.append([]);
        for j in range (0, M+1):
            return_matr[i].append(X[i]**j)
    return return_matr    

def computeY(W, X):
    Y = []
    for x_i in X:
        y_i = 0
        for i in range(0, len(W)):
            y_i = y_i + W[i]*((x_i)**(i))
        Y.append(y_i)
    return Y

def computeError(Y_target, Y_pred):
    error = 0
    for i in range(len(Y_target)):
        error = error + ((Y_target[i] - Y_pred[i])**2)
    error = error/len(Y_target)
    return error

def fitPoly(M):
    with open("C:\datafile\data1.pickle", "rb") as f:
        dataTrain,dataTest = pickle.load(f, encoding='latin1')
        
    X_train = []
    Y_train = []
    for i in range (len(dataTrain)):
        X_train.append(dataTrain[i][0])
        Y_train.append(dataTrain[i][1])     
        
    #A = numpy.vstack([x, numpy.ones(len(x))]).T
    Z = dataMatrix(X_train, M)
    result = numpy.linalg.lstsq(Z, Y_train)
    W = result[0]
    
    X_test = []
    Y_test = []
    for i in range (len(dataTest)):
        X_test.append(dataTest[i][0])
        Y_test.append(dataTest[i][1])
        
    Y_pred_train = computeY(W, X_train)
    Y_pred_test = computeY(W, X_test)
    
    err_train = computeError(Y_train, Y_pred_train)
    err_test = computeError(Y_test, Y_pred_test)
    
    return W, err_train, err_test
'''
fitPoly(4)
'''

def plotPoly(w):
    x = numpy.linspace(0, 1, num=1000)
    y = computeY(w,x)

    with open("C:\datafile\data1.pickle", "rb") as f:
        dataTrain,dataTest = pickle.load(f, encoding='latin1')
    
    X = []
    for i in range (len(dataTrain)):
        X.append(dataTrain[i][0])
    
    Y = []
    for i in range (len(dataTrain)):
        Y.append(dataTrain[i][1])
    
    plt.plot(X, Y, 'o', label='Original data', markersize=10)
    plt.plot(x, y, 'r', label='Fitted line')
    plt.legend()
    plt.show()
'''
W = fitPoly(4)[0]
plotPoly(W)
'''

def bestPoly():
    M_lst = []
    err_train_lst = []
    err_test_lst = []
    for i in range(16):
        M_lst.append(i)
        W, err_train, err_test = fitPoly(i)
        err_train_lst.append(err_train)
        err_test_lst.append(err_test)
        #plotPoly(result[0])
    
    plt.plot(M_lst, err_train_lst, 'o', label='Training error', markersize=8)
    #plt.plot(M_lst, err_test_lst, 'o', label='Test error', markersize=8)
    plt.legend()
    plt.show()
        
bestPoly()   



#Question 4
###############################################################################

def fitRegPoly(M, alpha):
    with open("data1.pickle", "rb") as f:
        dataTrain,dataTest = pickle.load(f, encoding='latin1')
    
    with open("data2.pickle", "rb") as f:
        dataVal,dataTest = pickle.load(f, encoding='latin1')
    X_val = []
    Y_val = []
    for i in range (len(dataVal)):
        X_val.append(dataVal[i][0])
        Y_val.append(dataVal[i][1])
    Z = dataMatrix(X_val, M) 

    ridge = lin.Ridge(alpha)
    ridge.fit(Z, Y_val)
    W = ridge.coef_
    W[0] = ridge.intercept_
    
    X_test = []
    Y_test = []
    for i in range (len(dataTest)):
        X_test.append(dataTest[i][0])
        Y_test.append(dataTest[i][1])        
    
    Y_pred_val = computeY(W, X_val)
    Y_pred_test = computeY(W, X_test)
    
    err_val = computeError(Y_val, Y_pred_val)
    err_test = computeError(Y_val, Y_pred_test)
    
    return W, err_val, err_test

def bestRegPoly():
    pass



#Question 5
###############################################################################
def regGrad(Z, t, w, alpha): 
    pass    

def fitPolyGrad(M, alpha, lrate):
    pass





























   