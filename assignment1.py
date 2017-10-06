#reycled from: https://artemrudenko.wordpress.com/2013/04/11/python-mxp-matrix-a-an-pxn-matrix-bmultiplication-without-numpy/
from itertools import product
import random


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



def mymeasure(I, K, J):
    '''
    :param I:
    :param K:
    :param J:
    :return:
    '''

    #I x K matrix:
    rows, cols = I, K
    matrix1 = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(random.randint(-999, 999))
        matrix1.append(col)

    #K x J matrix:
    rows, cols = K, J
    matrix2 = []
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(random.randint(-999, 999))
        matrix2.append(col)



print(mymeasure(2,3,4))

