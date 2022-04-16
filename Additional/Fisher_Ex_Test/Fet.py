"""
URL_GUIDE_1 = https://www.youtube.com/watch?v=udyAvvaMjfM
Scipy_DOC_FOR Fet = https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
Code_Example = https://www.statology.org/fishers-exact-test-python/

URL_GUIDE_2 = https://www.youtube.com/watch?v=8nm0G-1uJzA
"""
#Kinda weird example to be honest, in Scipy doc described more detailed instruction
data = [[8,4],
        [4,9]]

import scipy.stats as stats

print(stats.fisher_exact(data))