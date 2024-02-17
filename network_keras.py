from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import SGD
import numpy as np
## Se importan los modulos de keras necesarios


class Network(object):
    def __init__(self, sizes, momentum=0.0):
        self.model = Sequential()
        ## Se inicia una instancia de un modelo secuencial 
        self.model.add(Dense(units=sizes[1], activation='sigmoid', input_dim=sizes[0]))
        ## Se agrega la capa densa, con sizes[1] neuronas y dimensión sizes[0], f sigmoide
        for size in sizes[2:]:
            self.model.add(Dense(units=size, activation='sigmoid'))
            ## Hace lo mismo para las demas capas 
        optimizer = SGD(learning_rate=0.01, momentum=momentum)
        ## Se setea el optimizador con momentum
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        ## Se copila el modelo, con los optimizador sgd, función de perdida mean_squared_error y metrica accuracy

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    ## Método para el SGD usando fit
        x_train, y_train = zip(*training_data)
        ## Desempaquetando datos
        x_train, y_train = np.array(x_train), np.array(y_train)
        ## Convirtiendo a numpy
        if test_data: ## Si test_data no está vacio
            x_test, y_test = zip(*test_data)
            ## Se desempaquetan las tuplas
            x_test, y_test = np.array(x_test), np.array(y_test)
            ## Traduciendo a numpy
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=mini_batch_size, validation_data=(x_test, y_test))
            ## Se usa el método fit en el modelo usando el descenso estocástico 
    
    def evaluate(self, test_data):
        x_test, y_test = zip(*test_data)
        ## Desempaqueta los datos de entrada en dos listas
        x_test, y_test = np.array(x_test), np.array(y_test)
        ## Traduciendo a numpy
        return self.model.evaluate(x_test, y_test) 
        ## Evalua el rendimiendo del conjunto de prueba a través del modelo

    def predict(self, training_data):
        input_data = np.array(training_data)
        ## Convierte training en un arreglo numpy
        return self.model.predict(training_data)
        ## Realiza predicciones en los datos de entrada

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ## Carga los datos de mnist para los conjuntos test y train
    x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0
    ## Convierten a las imagenes en un array y los divide entre 255 para escalarlos
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    ## Usan to_categorical para convertir a representación one-hot encoding
    return (x_train, y_train), (x_test, y_test)
    ## Devuelve los conjuntos 

# Cargar datos MNIST
(x_train, y_train), (x_test, y_test) = load_mnist_data()

# Crear y entrenar la red neuronal
sizes = [784, 128, 10]  # Número de neuronas en cada capa y numero de capas
momentum = 0.95
## Valor del momentum
network = Network(sizes, momentum=momentum)
## Se crea una instancia de network
network.SGD(zip(x_train, y_train), epochs=29, mini_batch_size=32, eta=0.1, test_data=zip(x_test, y_test))
## Se llama a  la función SGD
## Se evalua el modelo al llamar a la función evaluate
accuracy = network.evaluate(zip(x_test, y_test))[1]
print(f'Precisión: {accuracy}%')
## Se imprime la precisión
