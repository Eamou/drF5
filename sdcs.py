import numpy as np

class SDCS:
    def __init__(self, n, k, M, A):
        self.n = n
        self.k = k
        self.M = M
        self.A = A
        self.s = [-1, 0, 1]
        self.z_M = [i for i in range(M)]
        self.table = self.gen_table()
    
    def gen_table(self):
        # for each b in z_M, exists some vector S s.t. a0.s0 + a1.s1 + ... = b with a in A
        sums = {i: list() for i in self.z_M}
        for i_val in range(-1,2):
            for j_val in range(-1, 2):
                for i in self.A:
                    for j in self.A:
                        if j > i:
                            sum = self.add(self.prod(i_val, i), self.prod(j_val, j))
                            sums[sum].append([i_val, j_val])
        # having generated all possible combinations, remove invalid ones
        # e.g, k limit for non-zero s values
        for i in sums:
            b = np.array(sums[i])
            remove_list = list()
            for s_pair in b:
                zero_count = np.count_nonzero(s_pair==0)
                if len(s_pair) - zero_count > self.k:
                    remove_list.append(s_pair)
            b = [list(x) for x in b]
            remove_list = [list(y) for y in remove_list]
            sums[i] = [x for x in b if x not in remove_list]
        return sums
            
    def add(self, num1, num2):
        return (num1 + num2) % self.M
    
    def prod(self, num1, num2):
        return (num1 * num2) % self.M

    def extract(self, sequence):
        # sum = a0.x0 + a1.x1 + ... + an.xn
        if len(sequence) > self.n:
            raise ValueError('Sequence too long')
        else:
            sum = 0
            for i in range(self.n):
                sum = self.add(sum, self.prod(self.A[i], sequence[i]))
            return sum % self.M
    
    def embed(self, x, b):
        # we are solving the equation a0.x0 + a0.s0 + a1.x1 + a1.s1 + ... + an.xn + an.sn = b
        ext_x = self.extract(x)
        desired = self.add(b, -ext_x)
        delta = self.table[desired]
        if len(delta) == 1:
            return delta[0]
        else:
            # try to return without -1 as it requires less bit changes
            for d in delta:
                if -1 not in d:
                    return d

test = SDCS(2, 1, 4, [1,2])
print(test.embed([1,1], 0))