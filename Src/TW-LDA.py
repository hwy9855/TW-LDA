from Src import DataPreProcessing
from Src import LDA

def main():
    dpre = DataPreProcessing.preprocessing('../Data/Dataset/Camera/Camera.docs')
    a = LDA.StandardLDA(dpre, 1, 0.1, 20, 1)
    a.train()
    print(a.phi)

def bcd():
    pass


if __name__ == '__main__':
    main()
