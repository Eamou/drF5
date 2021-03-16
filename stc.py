import numpy as np

class stc:
    def __init__(self, x, msg, H_hat):

        self.x = x
        self.msg = msg
        self.H_hat = H_hat

        self.h = len(bin(max(self.H_hat))[2:])
        self.w = np.shape(self.H_hat)[0]
        self.m = len(self.msg)
        self.n = len(self.x)
        self.bin_length = '0' + str(self.h) + 'b'

        self.rho = lambda x: 1 # this is the embedding cost for F5
        self.num_blocks = len(self.msg)

    def gen_H(self):
        H_hat_bin = np.array([list(format(x, self.bin_length))[::-1] for x in self.H_hat], dtype=np.uint8).T
        H = np.zeros((self.m,self.n))
        i, j = 0, 0
        i_step, j_step = 1, self.w
        while j < self.n:
            for mini_row in range(self.h):
                try:
                    H[i+mini_row][j:j+j_step] = H_hat_bin[mini_row]
                except:
                    break
            #H[i][j:j+j_step], H[i+i_step][j:j+j_step] = H_hat_bin
            i += i_step
            j += j_step
            if i == self.m-1:
                H[i][j:j+j_step] = H_hat_bin[0]
                return H

    def backward_viterbi(self, weights, path, x_index, m_index):
        # http://dde.binghamton.edu/filler/pdf/Fill10spie-syndrome-trellis-codes.pdf
        y = np.zeros(len(path))
        embedding_cost = int(min(weights))
        state = list(weights).index(embedding_cost) # 1
        x_index-=1
        m_index-=1 #code.n = num columns
        for _ in range(self.num_blocks, 0, -1):
            state = 2*state + self.msg[m_index] # this is the fix from the pseudocode
            m_index -= 1
            for j in range(self.w-1, -1, -1):
                y[x_index] = path[x_index][state]
                state = state ^ (int(y[x_index]*self.H_hat[j]))
                x_index -= 1
        return y, embedding_cost

    def generate(self):
        # forward viterbi
        # http://dde.binghamton.edu/filler/pdf/Fill10spie-syndrome-trellis-codes.pdf
        weights = np.array(np.ones(2**(self.h)) * np.inf)
        weights[0] = 0
        path = np.zeros((len(self.x), 2**(self.h)))
        x_index, m_index = 0, 0
        partial_syndromes = list()
        for s in range(2**self.h):
            partial_syndromes.append(int(format(s, self.bin_length)[::-1],2))
        for _ in range(self.num_blocks):
            for j in range(self.w):
                new_weights = np.array(np.zeros(2**(self.h)))
                for k in partial_syndromes:
                    w0 = weights[k] + (self.x[x_index] * self.rho(self.x[x_index]))
                    w1 = weights[k ^ self.H_hat[j]] + (1-self.x[x_index])*self.rho(self.x[x_index])
                    if w1 < w0:
                        new_weights[k] = w1
                        path[x_index][k] = 1
                    else:
                        new_weights[k] = w0
                        path[x_index][k] = 0
                x_index += 1
                weights = new_weights.copy()
            for j in range(2**(self.h-1)):
                weights[j] = weights[2*j + self.msg[m_index]]
            weights[2**(self.h-1):(2**self.h)] = np.inf
            m_index += 1
        return self.backward_viterbi(weights, path, x_index, m_index)

"""
H_hat = np.array([71,109], dtype=np.uint8)
m = np.array([0,1,1,1])
x = np.array([1,0,1,1,0,0,0,1])

stc_test = stc(x, m, H_hat)
H = stc_test.gen_H()
y, cost = stc_test.generate()
print(f"y: {y}")
assert np.array_equal((H @ y) % 2, m)
print("***pass***")
"""

#H = gen_H(H_hat, m, n)
#print(H)
#int(self.__conc_lst(list(self.H_hat.T[j][::-1])),2)
"""
def __conc_lst(self, arr):
        return ''.join([str(x) for x in arr])
"""