import numpy as np


# активационная функция
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# производная активационной функции
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# квадратичная целевая функция
# test_data - кортеж из входных значений и списка значений классов
def cost_function(network, test_data):
    c = 0
    for example, y in test_data:
        c += np.sum((network.feedforward(example) - y) ** 2)
    return c / (2 * len(test_data))


class Network:
    def __init__(self, shape, activation_function, activation_function_derivative, debug=True):
        self.shape = shape
        print(shape)
        self.biases = [np.random.randn(y, 1) for y in shape[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
        for i in range(0, 3):
            self.weights[i] = (self.weights[i] - 0.5) * 2 * np.sqrt(1 / shape[i])
            self.biases[i] = (self.biases[i] - 0.5) * 2 * np.sqrt(1 / shape[i])
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.debug = debug

    # прогнать до конца примеры из input_matrix
    def feedforward(self, input_matrix):
        for b, w in zip(self.biases, self.weights):
            # weigts - (массив матриц)
            input_matrix = self.activation_function(np.dot(w, input_matrix) + b)
        return input_matrix

    # обновление параметров нейронной сети (веса, смещения), сделав шаг градиентного спуска
    # на основе алгоритма обратного распространения ошибки
    # alpha - learning rate
    def back_prop_step(self, data, alpha):
        # значения dJ/db для каждого слоя
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # значения dJ/dw (ошибки) для каждого слоя
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # для каждого примера из батча применяем бек пропогейшн
        for x, y in data:
            delta_nabla_b, delta_nabla_w = self.back_prop_single_example(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = alpha / len(data)

        # обновляем параметры сети
        self.weights = [w - eps * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eps * nb for b, nb in zip(self.biases, nabla_b)]

    def prepare_data(data, classes_count):
        return [(row[:-classes_count].reshape(-1, 1), row[-classes_count:].reshape(-1, 1)) for row in data]

    def SGD(self, data_in, epochs, alpha, classes_count):
        errors = []

        for epoch in range(epochs):
            for row in data_in:
                input_data = row[:-classes_count].reshape(-1, 1)
                output_data = row[-classes_count:].reshape(-1, 1)
                self.back_prop_step([(input_data, output_data)], alpha)

            error = cost_function(self,
                                  [(row[:-classes_count].reshape(-1, 1), row[-classes_count:].reshape(-1, 1)) for row in
                                   data_in])
            if self.debug:
                print(f'epoch: {epoch} -  error:{error}')
                errors.append(error)
        return errors

    # возвращает вектор частных производных квадратичной целевой функции по активациям выходного слоя
    def cost_derivative(self, output_activations, y):
        return output_activations - y

    # алгоритм обратного распространения ошибки для одного примера из тренировочной выборки
    # возвращает кортеж (nabla_b, nabla_w) - градиентов по слоям по смещениям и весам соответственно
    def back_prop_single_example(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # прямое распространение (forward pass)
        # массив векторов активаций нейронов
        activations = [x]
        # массив векторов сумматорных функций от активаций предыдущих слоев

        summatories = []
        # b - вектор смещений нейронов данного слоя
        # w - матрица весов, входящих в данный слой
        for b, w in zip(self.biases, self.weights):
            summatories.append(np.dot(w, activations[-1]) + b)
            activation = self.activation_function(summatories[-1])
            activations.append(activation)

            # обратное распространение (backward pass)

            # ошибка для выходного слоя
        delta = self.cost_derivative(activations[-1], y) * self.activation_function_derivative(summatories[-1])
        # производная J по биасам выходного слоя
        nabla_b[-1] = delta
        # производная J по весам выходного слоя
        nabla_w[-1] = delta.dot(activations[-2].T)

        # Здесь l = 1 означает последний слой, l = 2 - предпоследний и так далее.
        for l in range(2, len(self.shape)):
            derivative = self.activation_function_derivative(summatories[-l])
            # ошибка на слое L-l
            delta = derivative * self.weights[-l + 1].T.dot(delta)
            # производная J по смещениям L-l-го слоя
            nabla_b[-l] = delta
            # производная J по весам L-l-го слоя
            nabla_w[-l] = delta.dot(activations[-l - 1].T)
        return nabla_b, nabla_w


def get_normalized_data(data, classes_count):
    normalized = (data / data.max())
    normalized.iloc[:, :-classes_count] = normalized.iloc[:, :-classes_count].fillna(0)
    mean = normalized.iloc[:, -classes_count:].mean()
    normalized.iloc[:, -classes_count:] = normalized.iloc[:, -classes_count:].fillna(mean)
    return normalized.to_numpy()


def rmse(network, train_data):
    return np.sqrt(network.cost_function(network, train_data))

