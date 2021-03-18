# https://stackoverflow.com/questions/7186518/function-with-varying-number-of-for-loops-python

import numpy as np

class sdcs:
    def __init__(self, params, A):
        self.n, self.k, self.M = params
        self.A = A
        self.s = [-1, 0, 1]
        self.z_M = [i for i in range(self.M)]
        self.table = {i: list() for i in self.z_M}
        self.gen_table()
    
    def loop_rec(self, val_dict, n):
        if n >= 1:
            for i_val in self.s:
                for i in self.A:
                    val_dict[n] = [i_val, i]
                    self.loop_rec(val_dict, n - 1)
        else:
            for j_val in self.s:
                for j in self.A:
                    val_dict[n] = [j_val, j]
                    if j > val_dict[n+1][1]:
                        sum = 0
                        for _, iter_var in val_dict.items():
                            sum = self.add(sum, self.prod(iter_var[0], iter_var[1]))
                        new_entry = [item[1][0] for item in val_dict.items()][::-1]
                        if new_entry not in self.table[sum]:
                            # we do this to avoid duplicates
                            self.table[sum].append(new_entry)
                        # need to reverse as the iterables are stored reversed compared to if these were normal nested for loops

    def gen_table(self):
        # perform n iterations of the nested for loops required to generate the values
        val_dict = {n: list() for n in range(self.n)}
        self.loop_rec(val_dict, self.n-1)
        # having generated all possible combinations, remove invalid ones
        # e.g, k limit for non-zero s values
        for i in self.table:
            for s_pair in self.table[i].copy():
                zero_count = np.count_nonzero(s_pair==0)
                if len(s_pair) - zero_count > self.k:
                    self.table[i] = np.delete(self.table[i], np.argwhere(self.table), axis=0)
            
    def add(self, num1, num2):
        # addition within finite field
        return (num1 + num2) % self.M
    
    def prod(self, num1, num2):
        # multiplication within finite field
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
            solns = list()
            for d in delta:
                xd_sum = [self.add(d[i], x[i]) for i in range(len(x))]
                if self.extract(xd_sum) == b:
                    solns.append(d)
            solns = np.array(solns)
            # now try to find the one with the least changes - sort by zero entries and return last
            return solns[(solns == 0).sum(axis=1).argsort()][-1]
                
def embedMsg(host, msg, sdcs):
    if len(host) != sdcs.n:
        raise Exception(f'Host array should be length n, where n specified by SDCS params: {sdcs.n}')
    else:
        if not all(x < sdcs.M for x in host):
            raise ValueError(f'All numbers in host array should be within the finite field specified by SDCS params: {sdcs.M}')
        else:
            int_msg = int(msg, 2)
            if int_msg > sdcs.M-1:
                raise ValueError(f'Message should be a number in the finite field Z_{sdcs.M}')
            else:
                delta = sdcs.embed(host, int_msg)
                return [host[i] + delta[i] for i in range(len(delta))]

#test = sdcs((3, 2, 17), [1,2,6])
#print(test.embed([1,0,1], 14))