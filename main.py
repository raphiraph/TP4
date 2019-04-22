import random
from random import randint
import keras
import numpy as np
from keras.datasets import imdb, reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

nn_param_choices = {
    'nb_neurons': [64, 128, 256, 512, 768, 1024],
    'activation': ['softmax', 'relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
}

generations = 5
population_size = 25

seed = 20
max_words = 10000

retain = 0.2
random_select = 0.2
mutate_chance = 0.2

early_stopper = EarlyStopping(patience=5)

def create_population(population_size):
    population = []
    for _ in range(0, population_size):
        neural_network = {"accuracy": 0., "network":{}}
        for key in nn_param_choices:
            neural_network["network"][key] = random.choice(nn_param_choices[key])
        population.append(neural_network)
    return population

def train_population(population):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_words)
    num_classes = np.max(y_train) + 1
    tokenizer = Tokenizer(num_words=max_words)
    X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    histories = []

    for neural_network in population:
        keras_model = create_keras_model(neural_network, num_classes)
        print(neural_network)
        history = keras_model.fit(X_train, y_train,
          batch_size=128,
          epochs=100,
          verbose=2,
          validation_data=(X_test, y_test),
          callbacks=[early_stopper])
        score = keras_model.evaluate(X_test, y_test, verbose=0)
        if neural_network["accuracy"] == 0.:
            neural_network["accuracy"] = score[1]
        histories.append(history)

    return histories

def create_keras_model(neural_network, num_classes):

    num_neurons = neural_network["network"]['nb_neurons']
    activation = neural_network["network"]['activation']
    optimizer = neural_network["network"]['optimizer']

    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def mutate(neural_network):

    mutation = random.choice(list(nn_param_choices.keys()))
    neural_network["network"][mutation] = random.choice(nn_param_choices[mutation])

    return neural_network

def breed(mother, father):
    babies = []
    for _ in range(2):

        baby_parameters = {}

        for param in nn_param_choices:
            baby_parameters[param] = random.choice(
                [mother[1]["network"][param], father[1]["network"][param]]
            )

        baby = {"accuracy": 0., "network": baby_parameters}

        if mutate_chance > random.random():
            baby = mutate(baby)

        babies.append(baby)

    return babies

def evolve(population):
    grades = [(neural_network["accuracy"], neural_network) for neural_network in population]
    sorted_grades = sorted(grades, key=lambda k: k[0], reverse=True)

    retain_length = int(len(sorted_grades)*retain)
    parents = sorted_grades[:retain_length]

    for neural_network in sorted_grades[retain_length:]:
        if random_select > random.random():
            parents.append(neural_network)

    parents_length = len(parents)
    desired_length = len(population) - parents_length
    children = []

    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)

        if male != female:
            male = parents[male]
            female = parents[female]
            babies = breed(male, female)

            for baby in babies:
                if len(children) < desired_length:
                    children.append(baby)

    parents.extend(children)
    return parents

def main():
    history_graph = []
    for i in range(generations):
        print("Generation {}".format(str(i)))
        population = create_population(population_size)
        histories = train_population(population)
        history_graph.append(histories)
        if i != generations - 1:
            population = evolve(population)
    population = sorted(population, key=lambda k: k["accuracy"], reverse=True)
    for neural_network in population[:5]:
        print(neural_network)
        print("Network accuracy: %.2f%%" % (neural_network["accuracy"] * 100))

    # Accuracy
    i=1
    for histories in history_graph:
        for j in range(population_size):
            plt.plot(histories[j].history['val_acc'])
        plt.title('Generation {} validation accuracy'.format(i))
        plt.ylabel('validation accuracy')
        plt.xlabel('epoch')
        plt.show()
        i+=1

    # Loss
    i=1
    for histories in history_graph:
        for k in range(population_size):
            plt.plot(histories[k].history['val_loss'])
        plt.title('Generation {} validation loss'.format(i))
        plt.ylabel('validation loss')
        plt.xlabel('epoch')
        plt.show()
        i+=1

if __name__ == '__main__':
    main()
