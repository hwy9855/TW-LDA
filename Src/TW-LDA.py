from Src import DataPreProcessing
from Src import LDA
import numpy as np


def init():
    dpre = DataPreProcessing.preprocessing('../Data/Dataset/Camera/Camera.docs')
    return dpre

def Stdlda(dpre):
    print("Start Step1 :")
    ldaModel = LDA.StandardLDA(dpre, 1, 0.1, 20, 1000)
    ldaModel.train(dpre)
    ldaModel.saveresult()
    print("Step1 Done\n")
    return ldaModel

def _bcd(ldaModel):
    print("Start Step2 :")
    ldaModel._phidot()
    bcd = np.zeros(ldaModel.dpre.words_count)
    for i in range(ldaModel.dpre.words_count):
        tmp = 0
        for x in range(ldaModel.K):
            tttmp = 0
            for y in range(ldaModel.K):
                tttmp += ldaModel.phidot[y][i]
            if ldaModel.phidot[x][i] != 0 and tttmp != 0:
                tmp += (ldaModel.phidot[x][i] / tttmp) * np.log2(ldaModel.phidot[x][i] / tttmp)
        tmp = 1 + tmp / np.log2(ldaModel.K)
        bcd[i] = tmp
    np.savetxt('../Res/bcd.txt', bcd)
    print("Step2 Done\n")
    return bcd

def des(dpre, bcd):
    print("Start Step3 :")
    for i in range(dpre.words_count):
        dpre.tw[i] = bcd[i]
    print("Step3 Done\n")

def xlda(dpre):
    print("Start Step4 :")
    xlda = LDA.StandardLDA(dpre, 1, 0.1, 20, 1500)
    xlda.train(dpre)
    xlda.saveresult()
    print("Step4 Done\n")
    return xlda

def main():
    dpre = init()
    ldaModel = Stdlda(dpre)
    # Step1
    bcd = _bcd(ldaModel)
    # Step2
    des(dpre, bcd)
    # Step3
    xlda(dpre)
    # Step4

if __name__ == '__main__':
    main()
