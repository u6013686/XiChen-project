from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
from keras.models import model_from_json
import numpy as np
import os
import re
def train_model(pattern):
    epochs = 5
    batch_size = 50
    print('Loading data')
    x, y, vocabulary, vocabulary_inv = load_data(pattern)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    sequence_length = x.shape[1] # 56
    vocabulary_size = len(vocabulary_inv) # 18765
    embedding_dim = 128
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.5

    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("Traning Model...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_validation, y_validation))  # starts training

    model.summary()

    model_json = model.to_json()
    with open("model" + pattern + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model" + pattern + ".h5")
    print("Saved model to disk")
    #np.savetxt(pattern + 'X_test.txt', X_test)
    #np.savetxt(pattern + 'y_test.txt', y_test)

    dirs = os.listdir(os.getcwd())
    max_ind = 0
    for file in dirs:
        if 'weights.00' in file:
            name_ind = int(re.search(r'weights\.00(.*?)-', file).group(1))
            if name_ind > max_ind:
                max_ind = name_ind
                weights_file = file
    # load json and create model
    # json_file = open('model' + pattern + '.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_file)
    print("Loaded model from disk")

    y_pred = [np.argmax(model.predict(np.array([X_test[i], ]))) for i in range(len(X_test))]

    def acc(y_true, y_pred):
        return np.equal(np.argmax(y_true, axis=-1), y_pred).mean()

    print("final test accuracy (optimal): " + str(acc(y_test, y_pred)))
    np.save(pattern + 'X_test.txt', X_test)
    np.save(pattern + 'y_test.txt', y_test)
    np.save(pattern + 'X_train.txt', X_train)
    np.save(pattern + 'y_train.txt', y_train)
def predict(pattern):
    # print(os.getcwd())
    print('Loading data')
    #vocabulary, vocabulary_inv = load_data_2(pattern)
    y = np.loadtxt(pattern +'y_test.txt')
    x = np.loadtxt(pattern +'X_test.txt')

    # n = 0
    # for i in range(len(x)):
    #     if y[i, 0] == 0:
    #         x = np.delete(x, i - n, axis=0)
    #         n += 1

    dirs = os.listdir(os.getcwd())
    max_ind = 0
    for file in dirs:
        if 'weights.00' in file:
            name_ind = int(re.search(r'weights\.00(.*?)-', file).group(1))
            if name_ind > max_ind:
                max_ind =name_ind
    weights_file = "weights.00"+ str(max_ind)+"-0.7547.hdf5"
    # load json and create model
    json_file = open('model'+pattern+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_file)
    print("Loaded model from disk")

    y_pred = [np.argmax(loaded_model.predict(np.array([x[i], ]))) for i in range(len(x))]

    def acc(y_true, y_pred):
        return np.equal(np.argmax(y_true, axis=-1),y_pred).mean()

    print("test accuracy: " + str(acc(y, y_pred)))
    #loaded_model.summary()