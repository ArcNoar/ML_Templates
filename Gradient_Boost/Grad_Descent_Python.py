
import time
start_time = time.time()
"""
url_of_guide = https://www.youtube.com/watch?v=JElfEE1OrSU
Url_of_text_guide = https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
"""

"""
HINTS:
1.Минимизируем функцию f(x)
2.Выбираем X_0 - Начальное приближение
3.Пусть X_n - Текущая найденная точка
4.Вычисляем градиент(производную)    f'(X_n)
5.Вычисляем следующее приближение :
	alpha = learning rate
	X_n+1 = X_n - alpha(f'(X_n))
"""

"""
We have:
X = {x_1,x_2,...,x_l} - Обучающая выборка
y = {y_1,y_2,...,y_l} - Целевая переменная (Действительное число)

"""

"""
Task:
Построить алгорится a(x), оптимизирующий функцию потерь.
 Q(y,a) = 1/l ( L ( sum ( y_i, a(x_i) )))

 Additional:
 Квадратичная функция потерь.
  L(y,a) = (a - y)**2
"""

"""
Чуть далее рассматривалось построение композиции, но я не понял что за c_i, и че мне с a_i делать
""" 

"""
Gradient descent in Python :
"""

cur_x = 3 # The algorithm starts at x=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*(x+5) #Gradient of our function 

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x+
    iters = iters+1 #iteration count
    print("Iteration",iters,"X value is",cur_x) #Print iterations
    
print("The local minimum occurs at", cur_x)



    

    

print(f'Время выполнения : {time.time() - start_time}')