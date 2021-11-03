import gslstats
import timeit
import numpy as np
import scipy.stats as stats

def ttest_all(data):
    ngps = data.shape[1]
    out = np.zeros((ngps, ngps))
    for i in range(ngps):
        for j in range(i+1, ngps):
            out[i, j] = stats.ttest_ind(data[:,i], data[:,j], equal_var=True).pvalue
            out[j, i] = out[i, j]
    return out

#with open('data.txt', 'r') as f:
#    data = np.array([[float(value) for value in line.strip().split(',')] for line in f.readlines()])

data = np.random.normal(size=(20,10))
print(data)
res = gslstats.ttest_all(data)
print(res)

#d1 = np.random.normal(size=(10,))
#d2 = np.random.normal(size=(10,))
#print(d1.flags.contiguous, d2.flags.contiguous);
#res = gslstats.ttest(d1, d2)
#print(res);
## Benchmarking
#t1 = timeit.timeit(
#        stmt='gslstats.ttest(d1, d2)', 
#        number=10000,
#        globals=globals(),
#    )
#t2 = timeit.timeit(
#        stmt='stats.ttest_ind(d1, d2, equal_var=True).pvalue', 
#        number=10000,
#        globals=globals(),
#    )
#print(t2/t1);
