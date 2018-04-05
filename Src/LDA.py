import numpy as np
import pandas as pd
import re

def input_data():
    tfmat = np.matrix(np.zeros([8958, 1230]))
    filename = '../Data/Dataset/Camera/Camera.docs'
    input = open(filename)
    V = 0
    for i in range(8958):
        tmp = input.readline()
        tmp = re.split(' |\n', tmp)
        tmp.pop()
        for token in tmp:
            tfmat[i, int(token)] += 1
            V += 1
    return tfmat, V

def main():
    V = 0
    K = 20
    alpha = 1
    beta = 0.1
    tfmat, V = input_data()
    print(tfmat, V)
    pass

if __name__ == '__main__':
    main()
