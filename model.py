from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
from keras.models import model_from_json
import numpy as np

def train_model(pattern):
    epochs = 5  #20
    batch_size =32  #64
    print('Loading data')
    x, y, vocabulary, vocabulary_inv = load_data(pattern)

    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
    np.savetxt(pattern+'X_train.txt', X_train)
    np.savetxt(pattern+'X_test.txt', X_test)
    np.savetxt(pattern+'y_train.txt', y_train)
    np.savetxt(pattern+'y_test.txt', y_test)
    
    sequence_length = x.shape[1] # 56
    vocabulary_size = len(vocabulary_inv) # 18765
    embedding_dim = 128 #256
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
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

    model.summary()

    model_json = model.to_json()
    with open("model" + pattern + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model" + pattern + ".h5")
    print("Saved model to disk")

def visualisation(pattern):
    print('Loading data')
    x, y, vocabulary, vocabulary_inv = load_data()

    # load json and create model
    json_file = open('model' + pattern + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model' + pattern + '.h5")
    print("Loaded model from disk")

    loaded_model.summary()
    from keras import backend as K
    inp = loaded_model.input  # input placeholder

    def generate_array(outputs):

        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
        layer_outs = functor(a)
        return np.asarray(layer_outs[0][0])

    def get_max_loc(outputs, filters):
        max_loc = []
        phrase_len = len(filters[:, 0, 0, 0])
        layer_out = generate_array(outputs)
        for i in range(512):
            max_l = np.argmax(layer_out[:, 0, i])
            max_loc.append((max_l, phrase_len))
        return max_loc

    import math
    num = 0

    for s in range(len(x)):
        pp = []
        pn = []
        num += 1
        print(num)
        sentence = [vocabulary_inv[e] for e in x[s]]
        a = [np.array([x[s], ]), 1.]
        outputs1 = []
        outputs2 = []
        outputs3 = []
        outputs4 = []
        outputs5 = []
        for layer in loaded_model.layers:
            # outputs.append(layer.output)
            if layer.get_config()['name'] == u'conv2d_1':
                outputs1.append(layer.output)
                filters1 = np.asarray(layer.get_weights()[0])
                # print (np.asarray(layer.get_weights()[0]).shape)
            elif layer.get_config()['name'] == u'conv2d_2':
                outputs2.append(layer.output)
                filters2 = np.asarray(layer.get_weights()[0])
            elif layer.get_config()['name'] == u'conv2d_3':
                outputs3.append(layer.output)
                filters3 = np.asarray(layer.get_weights()[0])
            elif layer.get_config()['name'] == u'dropout_1':
                outputs4.append(layer.output)
                layer_out4 = generate_array(outputs4)
            elif layer.get_config()['name'] == u'dense_1':
                filters5 = np.asarray(layer.get_weights()[0])  # evaluation function
                outputs5.append(layer.output)
                layer_out5 = generate_array(outputs5)

        locs = get_max_loc(outputs1, filters1) + get_max_loc(outputs2, filters2) + get_max_loc(outputs3, filters3)

        max_phrase = [0, None,None]
        for n in range(1536):
            pos_posibility = math.e ** ((filters5[n, 0]) * layer_out4[n]) / (
                        math.e ** ((filters5[n, 0]) * layer_out4[n]) + math.e ** ((filters5[n, 1]) * layer_out4[n]))
            if pos_posibility > max_phrase[0]:
                phrase = ' '.join(sentence[locs[n][0]:(locs[n][0] + locs[n][1] - 1)])
                max_phrase[0] = pos_posibility
                max_phrase[1] = phrase
                max_phrase[2] = num

        if layer_out5[1] > layer_out5[0]:
            pp.append(tuple(max_phrase))

    pp = sorted(pp, reverse=True)
    print('top 10 positive: ' + pp[0:9])