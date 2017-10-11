import numpy
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.linear_model as lin

#Question 1
#####################################################################################################
#Part (a)
def mymult(A, B):
    '''
    :param matr_a: The matrix A represented as lists of list
    :param matr_b: The matrix B represented as lists of list
    :return: return_matr: matrix to return after done mulplying A and N
    '''
    if len(A[0]) != len(B):
        return "The matrices cannot be multiplied"

    final_rows = len(A)
    final_cols = len(B[0])
    return_matr = [[0 for x in range(final_cols)] for y in range(final_rows)]

    for k in range (0, final_rows):
        for i in range (0, final_cols):
            value = 0
            for j in range (0, len(B)):
                value += A[k][j]*B[j][i]
            return_matr[k][i] = value

    return numpy.matrix(return_matr)
'''
matrixA = matrix([[1,2],[3,4],[5,6]]) 
matrixB = matrix([[7,8,9],[10,11,12]])
print(multiply(matrixA, matrixB))
'''

#Part (b)
def mymeasure(I, K, J):
    
    A = numpy.random.rand(I,K)    #I x K matrix
    B = numpy.random.rand(K,J)    #K x J matrix
    
    #compute time & compute the difference:
    start_time = time.time()
    C1 = numpy.matmul(A, B)
    finish_time = time.time()
    time_exe_matmul = finish_time - start_time
    print("numpy.matmul execution time: {0}".format(time_exe_matmul))
    
    start_time = time.time()
    C2 = mymult(A, B)
    finish_time = time.time()
    time_exe_mymult = finish_time - start_time
    print("mymult execution time: {0}".format(time_exe_mymult))
    
    magnitude_diff = numpy.sum(numpy.square(numpy.subtract(C1, C2)))
    print ("Magnitude of the difference between C1 and C2: {0}".format(magnitude_diff))
    
#Part (c)
'''  
mymeasure(1000,50,100)
mymeasure(1000,1000,1000)
'''

#Question 3
#####################################################################################################    
#Part (a)
def dataMatrix(X, M):
    return_matr = []
    for i in range (0, len(X)):
        return_matr.append([]);
        for j in range (0, M+1):
            return_matr[i].append(X[i]**j)
    return numpy.matrix(return_matr)    

#Part (b)
def fitPoly(M):
    with open("data1.pickle", "rb") as f:
        dataTrain,dataTest = pickle.load(f)
    
    X_train, Y_train = dataTrain[:,0], dataTrain[:,1]
    Z_train = numpy.vander(X_train, M+1, increasing=True)
    
    X_test, Y_test = dataTest[:,0], dataTest[:,1]
    Z_test = numpy.vander(X_test, M+1, increasing=True)
    
    result = numpy.linalg.lstsq(Z_train, Y_train)
    W = result[0]
    
    err_train = numpy.mean(numpy.square(numpy.subtract(Y_train, numpy.matmul(Z_train, W))))
    err_test = numpy.mean(numpy.square(numpy.subtract(Y_test, numpy.matmul(Z_test, W))))
    
    return W, err_train, err_test

#Part (c)
def plotPoly(w):
    x = numpy.linspace(0, 1, num=1000)
    y = numpy.matmul(numpy.vander(x, len(w), increasing=True), w)

    with open("data1.pickle", "rb") as f:
        dataTrain,dataTest = pickle.load(f)
    X, Y = dataTrain[:,0], dataTrain[:,1]

    plt.plot(X, Y, 'o', label='Original data', markersize=10)
    plt.plot(x, y, 'r', label='Fitted line')
    plt.ylim(-15,15)

#Part (d)
def bestPoly():
    M_lst = []
    err_train_lst = []
    err_test_lst = []
    
    for i in range(0,16):
        W, err_train, err_test = fitPoly(i)
        err_train_lst.append(err_train)
        err_test_lst.append(err_test)       
        M_lst.append(i)
        plt.subplot(4,4,i+1)
        plotPoly(W)
        
    plt.figure("Question3: training and test error")
    plt.plot(M_lst, err_train_lst, marker = 'o', color = 'blue', label='Training error', markersize=8)
    plt.plot(M_lst, err_test_lst, marker = 'o', color = 'red', label='Test error', markersize=8)
    plt.ylim(0,250)
    plt.legend()
    
    plt.figure("Question3: best-fitted polynomial (degree=4)")
    W, err_train, err_test = fitPoly(4)
    plotPoly(W)
    plt.legend()
    
    print("Optimal value of M: {0}".format(4))
    print("Optimal weight vector, w:\n{}".format(W))
    print("Training error: {}\nTest Error: {}".format(err_train, err_test))
'''
bestPoly()   
plt.show()
'''


#Question 4
#####################################################################################################
#Part (a)
def fitRegPoly(M, alpha):
    with open("data1.pickle", "rb") as f:
        dataTrain,dataTest0 = pickle.load(f)
    
    with open("data2.pickle", "rb") as f:
        dataVal,dataTest = pickle.load(f)
        
    X_train, Y_train = dataTrain[:,0], dataTrain[:,1]
    Z_train = numpy.vander(X_train, M+1, increasing=True)
          
    X_val, Y_val = dataVal[:,0], dataVal[:,1]
    Z_val = numpy.vander(X_val, M+1, increasing=True)
       
    ridge = lin.Ridge(alpha)
    ridge.fit(Z_train, Y_train)
    W = ridge.coef_
    W[0] = ridge.intercept_
    
    err_train = numpy.mean(numpy.square(numpy.subtract(Y_train, numpy.matmul(Z_train, W))))
    err_val = numpy.mean(numpy.square(numpy.subtract(Y_val, numpy.matmul(Z_val, W))))
    
    return W, err_train, err_val

#Part (b)
def bestRegPoly():
    with open("data2.pickle", "rb") as f:
        dataVal,dataTest = pickle.load(f)
        
    M = 15
    alpha_lst = []
    X_test, Y_test = dataTest[:,0], dataTest[:,1]
    Z_test = numpy.vander(X_test, M+1, increasing=True)
        
    
    err_val_lst, err_train_lst = [], []

    for i in range(-13, 3):
        alpha =  10**(i)
        W, err_train, err_val = fitRegPoly(M, alpha)
        
        alpha_lst.append(alpha)
        err_train_lst.append(err_train)
        err_val_lst.append(err_val)
        
        plt.subplot(4,4,i+13+1)
        plotPoly(W)

    plt.figure("Question 4: training and validation error")
    plt.semilogx(alpha_lst, err_train_lst, marker = 'o', color = 'blue', label='Training error', markersize=8)
    plt.semilogx(alpha_lst, err_val_lst, marker = 'o', color = 'red', label='Validation error', markersize=8)
    plt.ylim(0,250)
    plt.legend()
    
    plt.figure("Question4: best-fitting polynomial (alpha = 10^(-5))")
    W, err_train, err_val = fitRegPoly(M, 10**(-5))
    plotPoly(W)
    plt.legend()
    
    err_test = numpy.mean(numpy.square(numpy.subtract(Y_test, numpy.matmul(Z_test, W))))
    print("Optimal value of alpha: {0}".format(10**-5))
    print("Optimal weight vector, w:\n{}".format(W))
    print("Training error: {}\nValidation error: {}\nTest Error: {}".format(err_train, err_val, err_test))
'''
bestRegPoly()
plt.show()
'''



#Question 5
#####################################################################################################
#Part (c)
def regGrad(Z, t, w, alpha):
    to_mult = numpy.matmul(Z.T, numpy.subtract(t, numpy.matmul(Z, w)))
    d_lw = -2*to_mult
    d_reg = w*2*alpha
    d_reg[0] = 0
    
    return d_lw + d_reg

#Part (d)
def fitPolyGrad(M, alpha, lrate):
    with open("data1.pickle", "rb") as f:
        dataTrain,dataTest = pickle.load(f)
            
    X_train, Y_train = dataTrain[:,0], dataTrain[:,1]
    Z_train = numpy.vander(X_train, M+1, increasing=True)
    
    X_test, Y_test = dataTest[:,0], dataTest[:,1]
    Z_test = numpy.vander(X_test, M+1, increasing=True)
    
    num_iter = []
    err_train, err_test = [], []   
    
    w = numpy.random.rand(M+1)
    plot_ind = 1
    for i in range(10000000):
        w = numpy.subtract(w, numpy.array(lrate*regGrad(Z_train, Y_train, w, alpha)))
        if i in [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
            plt.subplot(3,3,plot_ind)
            plot_ind = plot_ind + 1
            plotPoly(w)
        if numpy.mod(i, 1000) == 0:
            num_iter.append(i)
            err_train.append(numpy.mean(numpy.square(numpy.subtract(Y_train, numpy.matmul(Z_train, w)))))
            err_test.append(numpy.mean(numpy.square(numpy.subtract(Y_test, numpy.matmul(Z_test, w)))))
            
    plt.figure("Training and test error vs time")
    plt.plot(num_iter, err_train, marker = 'o', color = 'blue', label='Training error', markersize=1)
    plt.plot(num_iter, err_test, marker = 'o', color = 'red', label='Test error', markersize=1)
    plt.legend()
    
    plt.figure("Question 5: fitted polynomial")
    plotPoly(w)
    plt.legend()
    
    print("Training error: {0}\nTest error: {1}\nweight vector: {2}".format(err_train[-1], err_test[-1], w))
'''
fitPolyGrad(15, 10**(-5), 0.02759)
plt.show()
'''


#***PLEASE UNCOMMENT THE FOLLWOING COMMANDS TO RUN THEM WHICH WAS USED TO WRITE REPORT***************
'''
print("********** Showing Question 1 ourtputs: **********\n")
mymeasure(1000,50,100)
#mymeasure(1000,1000,1000)

print("\n\n********** Showing Question 3 ourtputs: **********\n")
bestPoly()   
plt.show()

print("\n\n********** Showing Question 4 ourtputs: **********\n")
bestRegPoly()
plt.show()

print("\n\n********** Showing Question 5 outputs: **********\n")
fitPolyGrad(15, 10**(-5), 0.02759)
plt.show()
'''