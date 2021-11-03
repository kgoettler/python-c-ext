import gslstats
import timeit
import numpy as np
import scipy.stats as stats

d1 = list(np.random.normal(size=(10,)))
d2 = list(np.random.normal(size=(10,)))
res = gslstats.t_test_py(d1, d2)

# Benchmarking
t1 = timeit.timeit(
        stmt='gslstats.t_test_py(d1, d2)', 
        number=10000,
        globals=globals(),
    )
t2 = timeit.timeit(
        stmt='stats.ttest_ind(d1, d2, equal_var=True).pvalue', 
        number=10000,
        globals=globals(),
    )
print(t2/t1);
