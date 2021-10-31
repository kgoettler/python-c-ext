import gslstats
import timeit
import numpy as np
#import scipy.stats as stats

with open('data.txt', 'r') as f:
    data = np.array([[float(value) for value in line.strip().split(',')] for line in f.readlines()])
print(data)
gslstats.ttest_all(data)

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
