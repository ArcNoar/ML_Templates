"""
No_source , check link i example 1
"""
import numpy as np
import statsmodels.api as sm

x = np.arange(10).reshape(-1,1)
y = np.array([0,1,0,0,1,1,1,1,1,1])

x = sm.add_constant(x)


model = sm.Logit(y,x)

result = model.fit(method='newton')
print(result)

print(result.params)

print(result.predict(x))

print(result.predict(x) >= 0.5) # idk why this returns Boolean and my attempt to repeat code cause error

print(result.pred_table())

print(f'Some summary i guess??? : \n {result.summary()} \n Second Summary : \n {result.summary2()} ')

"""
                            Logit Regression Results
==============================================================================
Dep. Variable:                      y   No. Observations:                   10
Model:                          Logit   Df Residuals:                        8
Method:                           MLE   Df Model:                            1
Date:                Sat, 16 Apr 2022   Pseudo R-squ.:                  0.4263
Time:                        22:11:03   Log-Likelihood:                -3.5047
converged:                       True   LL-Null:                       -6.1086
Covariance Type:            nonrobust   LLR p-value:                   0.02248
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -1.9728      1.737     -1.136      0.256      -5.377       1.431
x1             0.8224      0.528      1.557      0.119      -0.213       1.858
==============================================================================
"""

"""
===============================================================
Model:              Logit            Pseudo R-squared: 0.426
Dependent Variable: y                AIC:              11.0094
Date:               2022-04-16 22:11 BIC:              11.6146
No. Observations:   10               Log-Likelihood:   -3.5047
Df Model:           1                LL-Null:          -6.1086
Df Residuals:       8                LLR p-value:      0.022485
Converged:          1.0000           Scale:            1.0000
No. Iterations:     7.0000
-----------------------------------------------------------------
          Coef.    Std.Err.      z      P>|z|     [0.025   0.975]
-----------------------------------------------------------------
const    -1.9728     1.7366   -1.1360   0.2560   -5.3765   1.4309
x1        0.8224     0.5281    1.5572   0.1194   -0.2127   1.8575
===============================================================
"""