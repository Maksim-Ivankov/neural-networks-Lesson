

# https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7
# урок 7
# Показатель ошибки - насколько неверна модель, которую мы пытаемсмся обучить
# котегориальная перекресная энатрапия в качестве показателя потерь

# одним из простых примеров функции потерь является средняя абсолютная ошибка, которая используется при регрессии


import math
softmax_output = [0.7,0.1,0.2]
target_output = [1,0,0]

loss = - (math.log(softmax_output[0])*target_output[0] + 
          math.log(softmax_output[1])*target_output[1] + 
          math.log(softmax_output[2])*target_output[2])

print(loss) # это потери, которые получили

# все, кроме 1 дают ноль при умноджении, пожтому формулу можно уупростить:
loss = -math.log(softmax_output[0])
print(loss)

# https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8
# Как включить показатель потерь в общую структуру нейронной сети
import numpy as np
import nnfs
from nnfs.datasets import spiral_data 

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

# Сделаем класс потерь
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategorialCrossentropy(Loss):
    def forward(self,y_pred,y_true): # закидываем расчетное значение выходного параметра нейронки и то, что ожидаем получить
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y  = spiral_data(samples = 100, classes = 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dence2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dence2.forward(activation1.output)
activation2.forward(dence2.output)
print('#################')
print(activation2.output[:5])

loss_function = Loss_CategorialCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)

# https://www.youtube.com/watch?v=txh3TQDwP1g&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=9
# Оптимизация весов и смещений
# урок 9

# Если изменять веса и смещения рандомно, то это бесмысслено. Потребуется слишком много времени, чтобы подобрать оптимальные значения. Вместо этого мы будем прибавлять 
# к прошлым весам и смещениям новые, и если потери уменьшаются продолжаем если нет, то просто берем новые веса и смещения и продоолжаем. 
# вот этот подход срабоатет и модель дейтсвительно обучится, понизив величину ошибки до меньше 20%































































