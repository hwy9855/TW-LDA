import numpy as np


class StandardLDA():
    def __init__(self, dpre, alpha, beta, K, iter):
        """
        :param dpre: data that have been preprocessed
        :param alpha: super parameter alpha
        :param beta: super parameter beta
        :param K: total number of topic
        :param iter: iteration times
        """
        self.dpre = dpre
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.iter = iter
        self.p = np.zeros(self.K)
        self.nw = np.zeros((self.dpre.words_count,self.K),dtype="int")
        self.nwsum = np.zeros(self.K,dtype="int")
        self.nd = np.zeros((self.dpre.docs_count,self.K),dtype="int")
        self.ndsum = np.zeros(dpre.docs_count,dtype="int")
        self.Z = np.array([ [0 for y in range(dpre.docs[x].length)] for x in range(dpre.docs_count)])

        for x in range(len(self.Z)):
            for y in range(self.dpre.docs[x].length):
                topic = np.random.randint(0,self.K-1)
                word = self.dpre.docs[x].words[y]
                tw = dpre.tw[word]
                self.Z[x][y] = topic
                self.ndsum[x] += tw
                self.nw[word][topic] += tw
                self.nd[x][topic] += tw
                self.nwsum[topic] += tw

        self.theta = np.array([ [0.0 for y in range(self.K)] for x in range(self.dpre.docs_count) ])
        self.phi = np.array([ [ 0.0 for y in range(self.dpre.words_count) ] for x in range(self.K)])
        self.phidot = np.array([ [ 0.0 for y in range(self.dpre.words_count) ] for x in range(self.K)])
        self.fc = np.zeros(self.K)

    def _theta(self):
        for i in range(self.dpre.docs_count):
            self.theta[i] = (self.nd[i]+self.alpha) / (self.ndsum[i]+self.K * self.alpha)

    def _phi(self):
        for i in range(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i]+self.dpre.words_count * self.beta)

    def _phidot(self):
        for x in range(self.dpre.docs_count):
            for y in range(self.dpre.docs[x].length):
                self.phidot[self.Z[x][y]][self.dpre.docs[x].words[y]] += 1
        for i in range(self.K):
            for j in range(self.dpre.words_count):
                self.phidot[i][j] = self.phidot[i][j] / self.fc[i]

    def Gibbs(self, dpre, i, j):

        topic = self.Z[i][j]
        word = self.dpre.docs[i].words[j]
        tw = dpre.tw[word]
        self.nw[word][topic] -= tw
        self.nd[i][topic] -= tw
        self.nwsum[topic] -= tw
        self.ndsum[i] -= tw

        Vbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha
        self.p = (self.nw[word] + self.alpha) / (self.nwsum + Kalpha) * (self.nd[i] + self.beta) / (self.ndsum[i] + Vbeta)
        #print(self.p)
        for k in range(1, self.K):
            self.p[k] += self.p[k-1]

        u = np.random.uniform(0,self.p[self.K-1])
        #if dpre.avgtw != 1:
        #    print(self.p, u)
        for topic in range(self.K):
            if self.p[topic]>u:
                break

        self.nw[word][topic] += tw
        self.nwsum[topic] += tw
        self.nd[i][topic] += tw
        self.ndsum[i] += tw

        return topic

    def saveresult(self):
        np.savetxt("../Res/theta.txt", self.theta, fmt='%.5f')
        np.savetxt("../Res/phi.txt", self.phi, fmt='%e')
        self.id2word = []
        with open('../Data/Dataset/Camera/Camera.vocab', 'r') as f:
            tmp = f.readlines()
            for line in tmp:
                line = line.split('\n')
                line.pop()
                line = str(line).split(':')
                line = line.pop()
                line = line.split('\'')
                line.pop()
                self.id2word.append(line)

        with open('../Res/topic_id.txt', 'w',) as f:
            for x in range(self.K):
                f.write('The ' + str(x) + ' class:\n')
                twords = []
                twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in range(self.dpre.words_count):
                    word = str(twords[y][0])
                    f.write('\t' + word + '\t' * 3 + str(twords[y][1]) + '\n')

        with open('../Res/tassgin.txt','w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y])+':'+str(self.Z[x][y]) + '\t')
                    self.fc[self.Z[x][y]] += 1
                f.write('\n')

    def train(self, dpre):
        for x in range(self.iter):
            for i in range(self.dpre.docs_count):
                for j in range(self.dpre.docs[i].length):
                    topic = self.Gibbs(dpre, i, j)
                    self.Z[i][j] = topic
            print(x)
        self._theta()
        self._phi()