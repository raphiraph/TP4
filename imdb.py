import random
from random import randint
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import numpy as np

# Initiate the random hyper parameters
nn_param_choices = {
    'nb_neurons': [5, 10, 25, 50, 75, 100],
    'activation': ['softmax', 'relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
}

# GA variables
generations = 5
population_size = 20
retain = 0.2
random_select = 0.1
mutate_chance = 0.2

# Dataset variables
seed = 20
top_words = 5000
max_words = 200

# Create a population of networks
def create_population(population_size):
    population = []
    for _ in range(0, population_size):
        neural_network = {"accuracy": 0., "network":{}}
        for key in nn_param_choices:
            # Assign random hyper parameters to each network
            neural_network["network"][key] = random.choice(nn_param_choices[key])
        population.append(neural_network)
    return population

# Train the population
def train_population(population):
    # Initialize the data set
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words, seed=seed)
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    # For graph purposes
    histories = []

    for neural_network in population:
        # Create the model
        keras_model = create_keras_model(neural_network)
        print(neural_network)
        # Train it
        history = keras_model.fit(X_train, y_train,
        batch_size=128,
        epochs=20,
        verbose=2,
        validation_data=(X_test, y_test))
        # Score it
        score = keras_model.evaluate(X_test, y_test, verbose=0)
        if neural_network["accuracy"] == 0.:
            neural_network["accuracy"] = score[1]
        # Save it
        histories.append(history)

    return histories

# Create a Keras model
def create_keras_model(neural_network):
    # Use the random hyper parameters
    num_neurons = neural_network["network"]['nb_neurons']
    activation = neural_network["network"]['activation']
    optimizer = neural_network["network"]['optimizer']

    model = Sequential()

    # We embed the first layer and flatten it
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Flatten())
    # The middle layer has a random number of neurons and a random activation function
    model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # The output layer has a random optimizer
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# A children's hyper parameter randomly mutates
def mutate(neural_network):
    mutation = random.choice(list(nn_param_choices.keys()))
    neural_network["network"][mutation] = random.choice(nn_param_choices[mutation])

    return neural_network

# Breed a child from two parents
def breed(mother, father):
    babies = []
    # We breed one child for every parent, therefore we breed two children
    for _ in range(2):

        baby_parameters = {}

        # Randomly select one of the parent's hyper parameter
        for param in nn_param_choices:
            baby_parameters[param] = random.choice(
            [mother["network"][param], father["network"][param]]
            )

        # Create a new network
        baby = {"accuracy": 0., "network": baby_parameters}

        # Some children will mutate
        if mutate_chance > random.random():
            baby = mutate(baby)

        babies.append(baby)

    return babies

# Determine who gets to survive
def evolve(population):
    # Rank networks by their accuracy
    sorted_population = sorted(population, key=lambda k: k["accuracy"], reverse=True)

    # How many networks will survive? Those who do get to become parents
    retain_length = int(len(sorted_population)*retain)
    parents = sorted_population[:retain_length]

    # We introduce a luck factor. Each discarded network has a chance to be "saved"
    for neural_network in sorted_population[retain_length:]:
        if random_select > random.random():
            parents.append(neural_network)

    # How many places are there to fill ?
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    children = []

    # While there is still some place to fill
    while len(children) < desired_length:
        # Randomly select a mother and father
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)

        # If they are not identical
        if male != female:
            male = parents[male]
            female = parents[female]
            # We breed them
            babies = breed(male, female)

            for baby in babies:
                if len(children) < desired_length:
                    children.append(baby)

    parents.extend(children)
    return parents

def main():
    history_graph = []
    # Create a population
    population = create_population(population_size)
    # For all generations
    for i in range(generations):
        print("Generation {}".format(str(i)))
        # Train the population
        histories = train_population(population)
        history_graph.append(histories)
        if i != generations - 1:
            # For all generation except the last, evolve
            population = evolve(population)
    # Rank the networks by accuracy
    population = sorted(population, key=lambda k: k["accuracy"], reverse=True)
    for neural_network in population[:5]:
        print(neural_network)
        print("Network accuracy: %.2f%%" % (neural_network["accuracy"] * 100))

    # Graph accuracy
    i=1
    for histories in history_graph:
        for j in range(population_size):
            plt.plot(histories[j].history['val_acc'])
        plt.title('Generation {} validation accuracy'.format(i))
        plt.ylabel('validation accuracy')
        plt.xlabel('epoch')
        plt.show()
        i+=1

if __name__ == '__main__':
    main()
