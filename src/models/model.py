import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPool2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.merge import concatenate


def Encoder(model_layers, input_shape):

    # initialize the sequential model
    model = Sequential(name="encoder")

    # add input layer
    model.add(Input(input_shape))

    # layers
    num_of_layers = len(model_layers)

    # build the network with the given parameters
    for layer in range(num_of_layers):

        # choose the model layer
        if (model_layers[layer][0] == "conv"):
            model.add( Conv2D(model_layers[layer][1], model_layers[layer][2], padding="same", activation="relu") )

        elif (model_layers[layer][0] == "pool"):
            model.add( MaxPooling2D(model_layers[layer][1]) )

        elif (model_layers[layer][0] == "batchNorm"):
            model.add( BatchNormalization() )

        elif (model_layers[layer][0] == "drop"):
            model.add( Dropout(model_layers[layer][1]) )


    return model


def Decoder(model_layers, input_shape):

    # initialize the sequential model
    model = Sequential(name="decoder")

    # add input layer
    model.add(Input(input_shape))

    # layers
    num_of_layers = len(model_layers)

    # build the network with the given parameters
    for layer in range(num_of_layers):

        # choose the model layer
        if (model_layers[layer][0] == "conv"):
            model.add( Conv2D(model_layers[layer][1], model_layers[layer][2], padding="same", activation="relu") )

        elif (model_layers[layer][0] == "pool" or model_layers[layer][0] == "upSample"):
            model.add( UpSampling2D(model_layers[layer][1]) )

        elif (model_layers[layer][0] == "batchNorm"):
            model.add( BatchNormalization() )
        
        elif (model_layers[layer][0] == "drop"):
            model.add( Dropout(model_layers[layer][1]) )

    return model

def Autoencoder(encoder_model, decoder_model):

    # merge both models into one
    autoencoder = Sequential(name="autoencoder")

    autoencoder.add( encoder_model.input )
    autoencoder.add(encoder_model)
    autoencoder.add(decoder_model)

    # add sigmoid layer
    autoencoder.add( Conv2D(1, (3, 3), activation='sigmoid', padding='same') )

    return autoencoder


def FullyConected(model_layers, input_shape):

    # initialize fully connected layer-model
    model = Sequential(name="fully_connected")

    # flat the input
    model.add( Input(input_shape) )
    model.add( Flatten() )
    
    # number of layers
    num_of_layers = len(model_layers)

    for layer in range(num_of_layers):

        # choose the fully connected model layer
        if (model_layers[layer][0] == "dense"):
            model.add( Dense(model_layers[layer][1], activation="relu") )

        elif (model_layers[layer][0] == "batchNorm"):
            model.add( BatchNormalization() )
        
        elif (model_layers[layer][0] == "drop"):
            model.add( Dropout(model_layers[layer][1]) )

    return model

def Classifier(encoder_model, fully_conected_model, classes):

    # merge both models into one
    classifier = Sequential(name="classifier")

    classifier.add( encoder_model.input )
    classifier.add(encoder_model)
    classifier.add(fully_conected_model)

    # add softmax layer
    classifier.add( Dense(classes, activation="softmax") )

    return classifier

def get_Autoencoder(model_info, input_shape):

    encoder = Encoder(model_info["encoder_layers"], input_shape)
    decoder = Decoder(model_info["decoder_layers"], encoder.output.get_shape()[1:])
    autoencoder = Autoencoder(encoder, decoder)
    
    # get the optimizer
    if (model_info["optimizer"][0] == "rmsprop"):
        optimizer = keras.optimizers.RMSprop(model_info["optimizer"][1])   
    elif (model_info["optimizer"][0] == "adam"):
        optimizer = keras.optimizers.Adam(model_info["optimizer"][1])   

    # compile the model with given hyperparameters
    autoencoder.compile(optimizer=optimizer,  loss="mean_squared_error", metrics=[ "accuracy", keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

    return autoencoder


if __name__ == "__main__":
    hamond_model = {"encoder_layers" : [["conv", 32, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 32, (3,3)],
                                        ["pool", (2,2)],
                                        ["conv", 64, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 64, (3,3)],
                                        ["pool", (2,2)],
                                        ["conv", 128, (3,3)],
                                        ["batchNorm"]]
                    ,
                    "decoder_layers" :  [["conv", 128, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 64, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 64, (3,3)],
                                        ["batchNorm"],
                                        ["upSample", (2,2)],
                                        ["conv", 32, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 32, (3,3)],
                                        ["batchNorm"],
                                        ["upSample", (2,2)]]
                    ,
                    "optimizer" :       ["adam", 0.01]
                    }

    stupid_model = {"encoder_layers" : [["conv", 32, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 32, (3,3)],
                                        ["pool", (2,2)],
                                        ["conv", 64, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 64, (3,3)],
                                        ["pool", (2,2)]]
                    ,
                    "decoder_layers" :  [["conv", 64, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 64, (3,3)],
                                        ["pool", (2,2)],
                                        ["conv", 32, (3,3)],
                                        ["batchNorm"],
                                        ["conv", 32, (3,3)],
                                        ["pool", (2,2)]]
                    ,
                    "optimizer" :       ["adam", 0.01]
                    }

    autoencoder = get_Autoencoder(hamond_model, [28,28,1])
    autoencoder.summary()