import numpy as np

class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

class DataPreProcessing(object):
    def __init__(self):
        self.words_count = 1230
        self.docs_count = 0
        self.docs = []
        self.tw = np.zeros(self.words_count)
        self.avgtw = 1


def preprocessing(filename):
    dpre = DataPreProcessing()
    print("input data.")
    input = open(filename)
    docs = input.readlines()
    input.close()
    dpre = DataPreProcessing()
    for line in docs:
        if line != "":
            tmp = line.strip().split()
            doc = Document()
            for item in tmp:
                doc.words.append(int(item))
            doc.length = len(tmp)
            dpre.docs.append(doc)
        else:
            pass
    dpre.docs_count = len(dpre.docs)
    for i in range(dpre.words_count):
        dpre.tw[i] = 1
    return dpre