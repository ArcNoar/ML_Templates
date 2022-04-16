"""
P1_URL_GUIDE(Brief Explanation) = https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLcVvgcJiXQ9r15zplwa1NDIB5SmUSoH3w&index=8&t=33s
P2_URL_GUIDE(Math_Details) = https://www.youtube.com/watch?v=2xudPOBz-vs
P3_URL_GUIDE(Classification) = https://www.youtube.com/watch?v=jxuNLH5dXCs
P4_URL_GUIDE(Math_Deatils) = https://www.youtube.com/watch?v=StWY5QWMXCw
"""
import numpy as np
import matplotlib.pyplot as plt

"""
P3 doodles. I'll just repeat some details for myself.


Init_Pred = np.log(4/2) # = log(Odds)
#print(Init_Pred) # = 0.7~ (0.6931)
# Elog = Log(4/2) but without log. just (4/2)


Some_Probability = np.exp(Init_Pred) / (1 + np.exp(Init_Pred)) # Equals Elog(4/2) / (1 + Elog(4/2))
print(Some_Probability) # = 0.7~ (0.6666)

Observed = 1
predicted = Some_Probability
Residual = (Observed - predicted)
#print(Residual) # 0.3 ~




plt.plot([Some_Probability,Some_Probability],'--',['No'],[0],'ro',['Yes'],[1],'bo')

plt.ylabel('Probability ')
plt.show()


# to calc new predict

Residual_I = -0.7 # Just an example 
Prev_Prob_I = Some_Probability
# In originale formula we use Sum of Residuals and Probabilities of leaf.

calc_form = Residual_I / (Prev_Prob_I * (1 - Prev_Prob_I))

#print(calc_form)


# Updating Weights
learning_rate = 0.8
Updater = Init_Pred + learning_rate * calc_form

New_Prob = np.exp(1.8) / (1 + np.exp(1.8)) # Equals e1.8 / (1 + e1.8)

print(New_Prob)
"""

"""
P4 Doodles.
"""
#Grad Boost Parts
#INPUT PART
"""
Data {(x_i,y_i)} n_iter ; i = index of data

Loss Function L(y_i,F(x)) = Instant Pred which equals mean of observed

Log(Likelihood of the Observed Data given the Prediction)  =
# ∑ = sum
for y_i in some_Y_array:
    sum(y_i * np.log(prob) + (1 - y_i) * np.log(1 - prob))


Sooo for example we will assume thath our pred probability(prob) is 0.67
and we have some y_i which will be : 0 - No , 1 - yes 
[0,1,1]
i will calculate this without iterations cause that will be easier to read.
"""
prob = 0.67

y_1_yes = 1
y_2_yes = 1
y_3_no = 0

first_Yes = y_1_yes * np.log(prob) + (1 - y_1_yes) * np.log(1 - prob) # Here it will be just np.log(0.67) = -0.4~
Second_Yes = y_2_yes * np.log(prob) + (1 - y_2_yes) * np.log(1 - prob) # Here it will be just np.log(0.67) = -0.4~
First_No = y_3_no * np.log(prob) + (1 - y_3_no) * np.log(1 - prob) # Here it will be just np.log(0.67) = - 1.1~

Original_Form = (first_Yes + Second_Yes + First_No) # = -1.9

"""
Negative Var
"""
#Formula = -(y * log(p) + (1 - y) * log(1-p))
"""
1) -(Observed * log(p) + (1 - Observed) * log(1 - p)
^
2) -Observed * log(p) - (1 - Observed) * log(1 - p)
^
3) -Observed * (log(p) - log(1 - p)) - log(1 - p)   #** Log(p) - Log(1 - p) = log(p) / log(1-p) = log(p/(1-p)) = Log(Odds)
^
4) -Observed * log(Odds) - log(1 - p) #** log(1 - p) => log(1 - (elog(Odds) / (1 + elog(Odds) ) )
                                      #***    ^ transforms into log( ( (1 + elog(odds) / (1 + elog(odds) ) -
                                      #                                                  ( elog(Odds) / ( 1 + elog(Odds) )
                                      #****  => log(1 / (1 + elog(Odds) ) )
                                      #***** => log(1) - log(1 + elog(Odds) )    ## Log(1) = 0 ; we can remove it
                                      #finnaly : -log(1 + elog(Odds) )

5) -Observed * log(Odds) + log(1 + elog(Odds))
It's our Loss Function
                                         
""" 
"""
Get the Derivative (d)

d / d(log(Odds)) | -Observed * log(Odds) + log(1 + elog(Odds)) 
                           ^                        ^
                      -Observed                 THE CHAAAAIN RULE
                                        +  1/(1 + elog(Odds)) * elog(Odds)    

=> - Observed + (elog(Odds) / (1 + elog(Odds) ) )
                               ^     ^      ^    
                        \       f(log(Odds))            /
                                    OR
                                we can use (p)

"""

# STEP 1 (Yeah. Only first step...)
 
"""
Observer  = y_i
Gamma = elog(Odds)

L(y_i,gamma) = Loss Function

Initialize model with a constant value : F_0(x) = argmin∑ L(y_i,gamma)

F_0(x) = log(yes_amount / no_amount) = init_pred
In reality there was some calculations, but i decided to skip that moment
"""

# STEP 2 

"""
Calculate residual for each sample in tree by getting (Observed -  last pred) 

To find Output Value for creating next tree
We aproximate calc with [Taylor Polynomial method]

Big Formula = 0

With power of coffeine ,The Product Rule and Chain Rule for second part of our calculations

only for first tree**
Gamma equal = Sum Of Residual / Sum of p * (1 - p)

for future trees F_m(X) updated based on first tree.

F_1(x) = F_0(x) + learning_Rate * (Leaf_Gamma)

in our case
(I forgot to mention about Root of first tree, i dont wanna to rewrite all conspect, so i'll just deal with it)
New Pred = 0.69 + 0.8 * 1.5 (Leaf_1_Yes)
New Pred = 0.69 + 0.8 * 0.7 (Leaf_2_Yes)
New Pred = 0.69 + 0.8 * 0.7 (Leaf_2_No)

"""

#STEP 3
"""
After all tree iterations. We use F_Final(X)
"""

