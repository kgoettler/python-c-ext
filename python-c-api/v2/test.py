import pandas as pd
import numpy as np
import scipy.stats as stats
IN_FILE = 'data.txt'
df = pd.read_csv(IN_FILE, delimiter=',', header=None)

#print(df.loc[:, [0,1]].agg(['mean', 'var']))
p = stats.ttest_ind(df[0].values, df[1].values).pvalue
print('Group 1 vs Group 2: p = {}'.format(p))
p = stats.ttest_ind(df[0].values, df[1].values, equal_var=False).pvalue
print('Group 1 vs Group 2: p = {}'.format(p))
p = stats.ttest_rel(df[0].values, df[1].values).pvalue
print('Group 1 vs Group 2: p = {}'.format(p))
