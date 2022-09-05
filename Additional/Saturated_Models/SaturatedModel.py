"""
URL_GUIDE = https://www.youtube.com/watch?v=9T0wlKdew6I
Deviance//
"""
# It's log(odds) values
LL_Null_Model = -3.51
LL_Proposed_Model = 1.27


R2_Except_SatM = (LL_Null_Model - LL_Proposed_Model) / LL_Null_Model
print(R2_Except_SatM) # = 1.36 (Wrong. It should be 0~1)

LL_Saturated_Model = 7.16

R2_With_SatM = (LL_Null_Model - LL_Proposed_Model) / (LL_Null_Model - LL_Saturated_Model)

print(R2_With_SatM) # = 0.44


"""
Deviance 
"""
Residual_Deviance = 2*(LL_Saturated_Model - LL_Proposed_Model)


"""
Double in this example makes the diff in log-Likelihoods with Degress Freedom equal to
the diff in the param_num
"""
print(Residual_Deviance) # = 11.78


Null_Deviance = 2*(LL_Saturated_Model - LL_Null_Model)

print(Null_Deviance) # = 21.34

"""
But in the end , Sat_Model in Log_Regress just equals zero, sooo... We don't need it...
"""