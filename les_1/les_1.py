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






































