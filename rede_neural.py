import tensorflow as tf
import keras as K

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D # type: ignore
from tensorflow.keras.layers import MaxPooling2D # type: ignore
from tensorflow.keras.layers import Flatten # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Inicializando a rede neural
classificador = Sequential()

# Primeira camada da convolução
classificador.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = 'relu'))

# Pooling do mapa gerado pela primeira camada de convolução
classificador.add(MaxPooling2D(pool_size=(2,2)))

# Segunda camada da convolução
classificador.add(Conv2D(32, (3,3), activation='relu'))

# Pooling do mapa gerado pela segunda camada de convolução
classificador.add(MaxPooling2D(pool_size=(2,2)))

# Realizando o flattening das informacoes em um vetor
classificador.add(Flatten())

# Juntando todas as camadas
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=1, activation='sigmoid'))

# Compilando a rede
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# PRÉ-PROCESSAMENTO DAS IMAGENS

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

dados_treino = ImageDataGenerator(rescale = 1./255, 
                                shear_range = 0.2, 
                                zoom_range = 0.2,
                                horizontal_flip = True)

dados_valid = ImageDataGenerator (rescale = 1./255)

elem_treino = dados_treino.flow_from_directory('/Users/jaolucena/Downloads/programacao/git/rede-neural-basica/dataframe/dataset_treino',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

elem_valid = dados_valid.flow_from_directory('/Users/jaolucena/Downloads/programacao/git/rede-neural-basica/dataframe/dataset_validacao',
                                             target_size = (64,64),
                                             batch_size = 32,
                                             class_mode = 'binary')

# TREINAMENTO DA REDE

classificador.fit_generator(dados_treino, 
                            steps_per_epoch = 8000,
                            epochs = 5,
                            dados_valid = elem_valid,
                            validation_steps = 2000)