# https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=1
# 3 входа в 1 нейрон
inputs = [1.2, 5.1, 2.1] # входные данные
weights = [3.1, 2.1, 8.7] # вес
bias = 3 # смещение
# первый шаг к созданию нейрона - сложить все входные данные, умноженные на весы, плюс смещение
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

# https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=2
# 4 входа в 3 нейрона
inputs = [1,2,3,2.5] # входные данные
weights1 = [0.2, 0.8, -0.5, 1.0] # вес
weights2 = [0.5, -0.91, 0.26, -0.5] # вес
weights3 = [-0.26, -0.27, 0.17, 0.87] # вес
bias1 = 2
bias2 = 3
bias3 = 0.5
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3,
          ]
          

print(output)
# задача глуюокого обучени - выяснить, как наилучшим образом настроить весы с учетом смещений, чтобы подобрать нужные нам выходные значения

# - https://www.youtube.com/watch?v=tMrbN67U9d4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=3
# оптимизация кода:
inputs = [1,2,3,2.5] # входные данные
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
baises = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, neuron_bais in zip(weights,baises):
    neuron_output = 0
    for n_input,weight in zip(inputs,neuron_weights):
        neuron_output+=n_input*weight
    neuron_output+=neuron_bais
    layer_outputs.append(neuron_output)
# код выше делает то же самое, что и на шаге 2, но в цикле, а не руками
print(layer_outputs)
# сейчас мы вводим руками веса и смещения, но дальше это бует делать автоматически оптимизатор
#--------------------------------------------------
# тензор - это объекьт, который может быть представлен в виде массива
# точечный подход:
import numpy as np

inputs = [1,2,3,2.5] # входные данные
weights = [0.2, 0.8, -0.5, 1.0]
baises = 2
output = np.dot(weights, inputs) + baises
print(output) # - получим такое же результат, как и раньше, но вычесления будут под капотом numpy - умножение скалярных векторов

# если передавать двумерные массивы, то это так же будет считаться правильно
inputs = [1,2,3,2.5] # входные данные
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
baises = [2, 3, 0.5]
output = np.dot(weights, inputs) + baises # здесь важно, что входные данные передаются вторым параметром
print(output)
# смещение с точки зрения функции активации поможет опрделить, активируется ли данный нейрон, и если да, то в какой степени

# https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4
# взять выборку входных данных и преобразовать в пакет входных данных
# так же до этого мы моделировали один слой нейронов, здесь смоделируем несколько слоев нейронов
# нейроны хорошо кладутся на масштабирование объектов в ООП, поэтому мы перейдем на ООП
# inputs = [[1,2,3,2.5],
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]] # входные данные
# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# baises = [2, 3, 0.5]
# output = np.dot(weights, inputs) + baises # здесь важно, что входные данные передаются вторым параметром
# print(output)
# если сейчас запустить код выше, то будет шибка формы, т.к. перемножение при перемножении матриц будет неправильная размерность.
# чтобы это починить, нужно выполнить транспонирование - поменять местами строки и столбцы в матрице весов
inputs = [[1,2,3,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] # входные данные
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
baises = [2, 3, 0.5]
output = np.dot(inputs, np.array(weights).T) + baises # здесь транспонируем (переворачиваем) матрицу weights, теперь входные данные должны идти первыми
print(output)
# далее сформируем второй слой

# ниже два слоя нейронов
inputs = [[1,2,3,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] # входные данные
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
baises = [2, 3, 0.5]
weights2 = [[0.1, -0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]
baises2 = [-1,2,-0.5]
layer1_output = np.dot(inputs, np.array(weights).T) + baises
layer2_output = np.dot(layer1_output, np.array(weights2).T) + baises2
print(layer2_output)
# но если писать таким образом, то очень мкоро кода станет настолько много, что он станет неуправляемым
# переходим к ООП
X = [[1,2,3,2.5], # X - стандарт в машинном обучении. Так обозначаютя входные параметры
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] # входные данные

class Layer_Dense:
    def __init__(self):
        pass
    
    def forward(self):
        pass
# задача дальше - преобразовать входые данные к диапозону от 0 до 1

np.random.seed(0) # исчесление от 0, каждый запуск дальше даст одно и то же рандомное значение

X = [[1,2,3,2.5], # X - стандарт в машинном обучении. Так обозначаютя входные параметры
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] # входные данные

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) # вес - рандом от размера входных данных и кол-ва нейронов
        #штука сверху возвращает это - [[ 1.76405235  0.40015721  0.97873798]
                                    #  [ 2.2408932   1.86755799 -0.97727788]
                                    #  [ 0.95008842 -0.15135721 -0.10321885]
                                    #  [ 0.4105985   0.14404357  1.45427351]]
        self.biases = np.zeros((1, n_neurons)) # без двойных скобок не работает!
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

print('----------------')

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# на данно мэтапе нам не хватает следующей важной вещи - функции активации
# https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
# Функции активации, Выпрямленная линейная функция, ступенчатая функция, сигмавиденая функция

# пошаговая функция - если от нуля, то 1

# сигмавидная функция - y=1/(1+e^-x) 
# Здесь получаем более детализированный выходной сигнал в диапазоне от 0 до 1
# более предпочтительна и надежна для обучения нейронной сети
# есть проблема исчезающего градиента

# линейная активация - если на выходе получаем больше нуля, то пускаем дальше сам x если меньше нуля - то ноль
# быстрый, но детализированый
# самая популярная функция активации скрытых слоев в нейронных сетях
print('----------------')
X = [[1,2,3,2.5], # X - стандарт в машинном обучении. Так обозначаютя входные параметры
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] # входные данные

inputs = [0.2,-1,3.3,-2.7,1.1,2.2,-100]
output = []

# линейная функция активации:
# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i<= 0:
#         output.append(0)

# ниже тоже самое, но короче
for i in inputs:
    output.append((max(0,i)))

print(output)

# функция активации на классах

np.random.seed(0) # исчесление от 0, каждый запуск дальше даст одно и то же рандомное значение

X = [[1,2,3,2.5], # X - стандарт в машинном обучении. Так обозначаютя входные параметры
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] # входные данные

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) # вес - рандом от размера входных данных и кол-ва нейронов
        self.biases = np.zeros((1, n_neurons)) # без двойных скобок не работает!
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: # здесь пишем фукнцию активации
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# пакет nnfs - установка через pip

import nnfs
from nnfs.datasets import spiral_data
nnfs.init() # устанавливает тип данных по умолчанию, который будет использоватьбся нампи

X,y = spiral_data(100,3) # данные для модели - координаты спиралей

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) # вес - рандом от размера входных данных и кол-ва нейронов
        self.biases = np.zeros((1, n_neurons)) # без двойных скобок не работает!
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: # здесь пишем фукнцию активации
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

#https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
# Урок 6 - Активация Softmax
# зачем нужна ещё одна функция активации, разве линейной недостаточно? На самом деле у нее есть проблема нуля. Если на каком-то слое возникнет выход меньше нуля, 
# линейная функция активации просто обрежет это значение до нуля. И до самого выходного слоя тут будет ноль.
# Для увеличения точнсти обучения модели, эти значения нельзя обрезать, а нужно учитывать. Функция Активация Softmax это учитывает

# Активация Softmax - y=e^x

import math

layer_outputs = [4.8,1.21,2.385]

E = math.e

exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)

# Нормализация данных - значение одного нейрона, деленное на сумму всех остальных выходных нейронов это неройна. Это дает нам желаемое распределение вероятностей

norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value/norm_base) # получаем распределения вероятностей от 0 до 1

print(norm_values)
# код выше можно упростить с помощью nampy:
# получиться гораздо короче
layer_outputs = [4.8,1.21,2.385]
exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)
print(norm_values)

# применим функцию активации Softmax к пакетам входных данных:


layer_outputs = [[4.8,1.21,2.385],
                 [8.9, -1.81, 0.2],
                 [1.41,01.051, 0.026]]
exp_values = np.exp(layer_outputs) # возводим в степень каждое число в входных данных по основанию e
# print(np.sum(layer_outputs, axis=1, keepdims=True)) # просуммирует строки отдельно, сохранит в виде матрицы вертикальной 1х3
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)
# [[8.95282664e-01 2.47083068e-02 8.00090293e-02]
#  [9.99811129e-01 2.23163963e-05 1.66554348e-04]    - вернет это
#  [5.13097164e-01 3.58333899e-01 1.28568936e-01]]

# при работе с e^x есть проблема, связанная с тем, что слишком быстро прирастает y, который в какой-то момент может просто переполниться. 
# Нужно решить проблему переполнения степени экспоненты

# решение - вычесть максимальное значение из всех и перемевернуть знак.

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) # вес - рандом от размера входных данных и кол-ва нейронов
        self.biases = np.zeros((1, n_neurons)) # без двойных скобок не работает!
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: # здесь пишем фукнцию активации
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,  axis=1, keepdims=True)) # эта штука предотвращает переполнение
        probabilites = exp_values / np.sum(exp_values,  axis=1, keepdims=True)
        self.output = probabilites


X, y  = spiral_data(samples = 100, classes = 3)

dense1 = Layer_Dense(2,3)
acivation1 = Activation_ReLU()

dence2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dence2.forward(activation1.output)
activation2.forward(dence2.output)
print('#################')
print(activation2.output[:5])






















































