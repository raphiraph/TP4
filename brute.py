from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

seed = 20
top_words = 20000
max_words = 2000

early_stopper = EarlyStopping(patience=5)

def train_population():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words, seed=seed)
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    keras_model = create_keras_model()
    history = keras_model.fit(X_train, y_train,
      batch_size=128,
      epochs=100,
      verbose=2,
      validation_data=(X_test, y_test),
      callbacks=[early_stopper])
    score = keras_model.evaluate(X_test, y_test, verbose=0)
    print("Results accuracy : {}".format(score[1]*100))
    print("Results loss : {}".format(score[0]))
    plt.plot(history.history['val_acc'])
    plt.title('Validation accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(history.history['val_loss'])
    plt.title('Validation loss')
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.show()

def create_keras_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train_population()
