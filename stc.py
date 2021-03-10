from functools import partial
import numpy as np

class stc:
    def __init__(self, x, msg, H_hat):

        self.x = x
        self.msg = msg
        self.H_hat = H_hat

        self.h, self.w = np.shape(self.H_hat)
        self.m = 2**(self.h)
        self.n = len(self.x)

        self.H = self.__gen_H()
        self.rho = lambda a: 1 # this is the embedding cost for F5
        self.num_blocks = np.shape(self.H)[1] // 2

    def __conc_lst(self, arr):
        return ''.join([str(x) for x in arr])

    def __gen_H(self):
        H = np.zeros((self.m,self.n))
        i, j = 0, 0
        i_step, j_step = 1, self.w
        while j < self.n:
            H[i][j:j+j_step], H[i+i_step][j:j+j_step] = H_hat
            i += i_step
            j += j_step
            if i == self.m-1:
                H[i][j:j+j_step] = H_hat[0]
                return H

    def backward_viterbi(self, weights, path, x_index, m_index):
        y = np.zeros(len(path))
        print(weights)
        embedding_cost = int(min(weights))
        state = list(weights).index(embedding_cost) # 1
        x_index-=1
        m_index-=1 #code.n = num columns
        for _ in range(self.num_blocks, 0, -1):
            state = 2*state + self.msg[m_index]
            m_index -= 1
            for j in range(self.w-1, -1, -1):
                print("pre y: ", state)
                y[x_index] = path[x_index][state]
                state = state ^ (int(y[x_index]*int(self.__conc_lst(list(self.H_hat.T[j][::-1])),2)))
                x_index -= 1
        return y, embedding_cost

    def forward_viterbi(self):
        weights = np.array(np.ones(2**(self.h)) * np.inf)
        weights[0] = 0
        path = np.zeros((len(self.x), 2**(self.h)))
        x_index, m_index = 0, 0
        bin_length = '0' + str(self.h) + 'b'
        partial_syndromes = list()
        for s in range(2**self.h):
            partial_syndromes.append(int(format(s, bin_length)[::-1],2))
        for _ in range(self.num_blocks):
            for j in range(self.w):
                #print("col:", weights, path)
                new_weights = np.array(np.zeros(2**(self.h)))
                for k in partial_syndromes:
                    w0 = weights[k] + (self.x[x_index] * self.rho(self.x[x_index]))
                    w1 = weights[k ^ int(self.__conc_lst(list(self.H_hat.T[j][::-1])),2)] + (1-self.x[x_index])*self.rho(self.x[x_index])
                    if w1 < w0:
                        new_weights[k] = w1
                        path[x_index][k] = 1
                    else:
                        new_weights[k] = w0
                        path[x_index][k] = 0
                x_index += 1
                weights = new_weights.copy()
            #print("prune:", weights, path)
            for j in range(2**(self.h-1)):
                weights[j] = weights[2*j + self.msg[m_index]]
            weights[2**(self.h-1):(2**self.h)] = np.inf
            m_index += 1
        print(path)
        return self.backward_viterbi(weights, path, x_index, m_index)

H_hat = np.array([[1,0],[1,1]])
m = np.array([0,1,1,1])
x = np.array([1,0,1,1,0,0,0,1])

stc_test = stc(x, m, H_hat)
y, cost = stc_test.forward_viterbi()
print("y: ", y)
print("message: ", (stc_test.H @ y)%2)
#H = gen_H(H_hat, m, n)
#print(H)