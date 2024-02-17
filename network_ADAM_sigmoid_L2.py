from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
## Se importan los modulos de keras necesarios


class Network(object):
    def __init__(self, sizes):
        self.model = Sequential()
        ## Se inicia una instancia de un modelo secuencial 
        self.model.add(Dense(units=sizes[1], activation='sigmoid', input_dim=sizes[0], kernel_regularizer=l2(0.02)))
        ## Se agrega la capa densa, con sizes[1] neuronas y dimensión sizes[0], f sigmoide, regularizador l2
        for size in sizes[2:-1]:
            self.model.add(Dense(units=size, activation='sigmoid', kernel_regularizer=l2(0.02)))
            ## Hace lo mismo para las demas capas 
        self.model.add(Dense(units=sizes[-1], activation = 'softmax'))
        ## Capa final con función de activación softmax
        optimizer =   Adam(learning_rate=0.0014, beta_1=0.93, beta_2=0.98)
        ## Se setea el optimizador ADAM
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        ## Se copila el modelo, con los optimizador ADAM, función de perdida mean_squared_error y metrica accuracy

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

    def predict(self, input_data):
        input_data = np.array(input_data)
        ## Convierte input_data en un arreglo numpy
        return self.model.predict(input_data)
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
sizes = [784, 512, 256, 256, 512, 1024, 10]  # Número de neuronas en cada capa y numero de capas
network =  Network(sizes)
## Se crea una instancia de network
network.SGD(zip(x_train, y_train), epochs=29, mini_batch_size=32, eta=0.1, test_data=zip(x_test, y_test))
## Se llama a  la función SGD
## Se evalua el modelo al llamar a la función evaluate
accuracy = network.evaluate(zip(x_test, y_test))[1]
print(f'Precisión: {accuracy}%')
## Se imprime la precisión
