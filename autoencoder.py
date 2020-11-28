
# import needed packages
import numpy as np
import os
import sys
import struct
import json
from array import array as pyarray
from keras.utils import normalize
# inport our files
from model import get_Autoencoder, train_Autoencoder
from visualization import autoencoder_visualization

# Define class with colors for UI improvement
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Define class for reading data from MNIST file
def load_mnist(dataset, digits=np.arange(10), type='data', numOfElements=-1):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    if not os.path.isfile(dataset):
        return None
    fname = os.path.join(".", dataset)
    if (type == 'data'):
        nMetaDataBytes = 4 * intType.itemsize
        images = np.fromfile(fname, dtype = 'ubyte')
        magicBytes, size, rows, cols = np.frombuffer(images[:nMetaDataBytes].tobytes(), intType)
        if numOfElements == -1:
            numOfElements = size #int(len(ind) * size/100.)
        images = images[nMetaDataBytes:].astype(dtype = 'float32').reshape([numOfElements, rows, cols, 1])
        return images
    elif (type == 'labels'):
        nMetaDataBytes = 2 * intType.itemsize
        labels = np.fromfile(fname, dtype = 'ubyte')[nMetaDataBytes:]
        return labels
    else:
        return None


# Define class for reading hyperparameters
def read_hyperparameters():
    validInput = False
    while not validInput:
        answer = input(bcolors.OKCYAN+'Do you want to import already existed hyperparameters\' configuration? (answer: y|n) '+bcolors.ENDC)
        if answer == 'y' or answer == 'Y' or answer == 'n' or answer == 'N':
            validInput = True
        else:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    if answer == 'y' or answer == 'Y':
        validInput = False
        while not validInput:
            confName = input(bcolors.OKCYAN+'Please add your configuration\'s path: '+bcolors.ENDC)
            if os.path.isfile(confName):
                with open(confName) as json_file:
                    try:
                        data = json.load(json_file)
                        existed_conf = True
                        model_info = data['model_info']
                        return model_info
                    except ValueError:
                        print(bcolors.FAIL+'Error: bad configuration file.'+bcolors.ENDC)
            else:
                print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
    # Define the dictionary with model's info
    model_info = {}

    # Number of convolutional layers on Encoder
    validInput = False
    while not validInput:
        numOfLayers = input(bcolors.OKCYAN+'Give number of convolutional layers on encoder: '+bcolors.ENDC)
        try:
            numOfLayers = int(numOfLayers)
            if numOfLayers > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

    # Layers of Encoder
    model_info['encoder_layers'] = list();
    for i in range(numOfLayers):
        # Type of ith convolutional layer
        validInput = False
        while not validInput:
            layer_type = input(bcolors.OKCYAN+'Choose type of layer number '+ str(i+1)+ ' (conv/batchNorm/pool): '+bcolors.ENDC)
            if layer_type == 'conv' or layer_type == 'batchNorm' or layer_type == 'pool':
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

        if layer_type == 'conv':
            # Number of filters of ith convolutional layer
            validInput = False
            while not validInput:
                numOfFilters = input(bcolors.OKCYAN+'Give number of convolutional filters for the layer : '+bcolors.ENDC)
                try:
                    numOfFilters = int(numOfFilters)
                    if numOfFilters > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            # Size of filters of ith convolutional layer
            validInput = False
            while not validInput:
                sizeOfFilters_str = input(bcolors.OKCYAN+'Give size of convolutional filters for the layer: '+bcolors.ENDC)
                try:
                    x, y = sizeOfFilters_str.split()
                    x = int(x)
                    y = int(y)
                    if x > 0 and y > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            model_info['encoder_layers'].append([layer_type, numOfFilters, (x, y)])

        elif layer_type == 'pool':
            # Size of filters of ith convolutional layer
            validInput = False
            while not validInput:
                sizeOfFilters_str = input(bcolors.OKCYAN+'Give size of convolutional filters for the layer: '+bcolors.ENDC)
                try:
                    x, y = sizeOfFilters_str.split()
                    x = int(x)
                    y = int(y)
                    if x > 0 and y > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            model_info['encoder_layers'].append([layer_type, (x,y)])

        else: # layer_type == 'batchNorm'
            model_info['encoder_layers'].append([layer_type])

    # Number of convolutional layers on Decoder
    validInput = False
    while not validInput:
        numOfLayers = input(bcolors.OKCYAN+'Give number of convolutional layers on decoder: '+bcolors.ENDC)
        try:
            numOfLayers = int(numOfLayers)
            if numOfLayers > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    
    # Layers of Decoder
    model_info['decoder_layers'] = list();
    for i in range(numOfLayers):
        # Type of ith convolutional layer
        validInput = False
        while not validInput:
            layer_type = input(bcolors.OKCYAN+'Choose type of layer number '+ str(i+1)+ ' (conv/batchNorm/upSample): '+bcolors.ENDC)
            if layer_type == 'conv' or layer_type == 'batchNorm' or layer_type == 'upSample':
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

        if layer_type == 'conv':
            # Number of filters of ith convolutional layer
            validInput = False
            while not validInput:
                numOfFilters = input(bcolors.OKCYAN+'Give number of convolutional filters for the layer: '+bcolors.ENDC)
                try:
                    numOfFilters = int(numOfFilters)
                    if numOfFilters > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            # Size of filters of ith convolutional layer
            validInput = False
            while not validInput:
                sizeOfFilters_str = input(bcolors.OKCYAN+'Give size of convolutional filters for the layer: '+bcolors.ENDC)
                try:
                    x, y = sizeOfFilters_str.split()
                    x = int(x)
                    y = int(y)
                    if x > 0 and y > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            model_info['decoder_layers'].append([layer_type, numOfFilters, (x, y)])

        elif layer_type == 'upSample':
            # Size of filters of ith convolutional layer
            validInput = False
            while not validInput:
                sizeOfFilters_str = input(bcolors.OKCYAN+'Give size of convolutional filters for the layer: '+bcolors.ENDC)
                try:
                    x, y = sizeOfFilters_str.split()
                    x = int(x)
                    y = int(y)
                    if x > 0 and y > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            model_info['decoder_layers'].append([layer_type, (x,y)])

        else: # layer_type == 'batchNorm'
            model_info['decoder_layers'].append([layer_type])

    # Number of epochs
    validInput = False
    while not validInput:
        epochs = input(bcolors.OKCYAN+'Give number of epochs: '+bcolors.ENDC)
        try:
            epochs = int(epochs)
            if epochs > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    model_info['epochs'] = epochs

    # Batch size
    validInput = False
    while not validInput:
        batch_size = input(bcolors.OKCYAN+'Give batch size: '+bcolors.ENDC)
        try:
            batch_size = int(batch_size)
            if batch_size > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
             print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    model_info['batch_size'] = batch_size

    # Type of activation function
    validInput = False
    while not validInput:
        af_type = input(bcolors.OKCYAN+'Choose type of activation function (Sigmoid/Linear/Relu): '+bcolors.ENDC)
        if af_type == 'Sigmoid' or af_type == 'Linear' or af_type == 'Relu':
            validInput = True
        else:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    model_info['activation_function'] = af_type

    # Model's optimizer
    model_info['optimizer'] = ['adam', 0.01]
    
    print(model_info)
    return model_info 


# Main Function
def main():
    print('argument list:', str(sys.argv))

    # Reading inline arguments
    if '-d' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-d>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' -d <dataset>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-d') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: missing value for argument <-d>.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' -d <dataset>'+bcolors.ENDC)
            sys.exit()
        datasetFile = sys.argv[sys.argv.index('-d')+1]

    # Reading dataset
    if not os.path.isfile(datasetFile):
        print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
        sys.exit()
    data = normalize(load_mnist(datasetFile))
    
    # Executer experiment
    histories = list()
    repeat = True
    while repeat:
        # Ask if there is already saved model
        validInput = False
        while not validInput:
            answer = input(bcolors.OKCYAN+'Do you want to import already existed model? (answer: y|n) '+bcolors.ENDC)
            if answer == 'y' or answer == 'Y' or answer == 'n' or answer == 'N':
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        if answer == 'y' or answer == 'Y':
            validInput = False
            while not validInput:
                model_info = input(bcolors.OKCYAN+'Please add your model\'s path: '+bcolors.ENDC)
                if os.path.isfile(model_info):
                    validInput = True
                else:
                    print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
        else:
            # Reading hyperparameters from user
            model_info = read_hyperparameters()
        
        autoencoder = get_Autoencoder(model_info, data.shape[1:])

        # Run Experiment!
        print(bcolors.BOLD+'\nTRAINING'+bcolors.ENDC)
        print(bcolors.BOLD+'----------------------------------------------------'+bcolors.ENDC)
        print(data.shape)
        histories.append(train_Autoencoder(autoencoder, model_info, data))

        # Check what user wants to do next
        endOfExperiment = False
        while not endOfExperiment:
            print(bcolors.BOLD+'\n----------------------------------------------------'+bcolors.ENDC+bcolors.OKCYAN)
            print('1. Repeat experiment with diferent hyperparameters.')
            print('2. Show graphs of error.')
            print('3. Save model.')
            print('4. Exit Program.')
            choice = input('Choose something from above: '+bcolors.ENDC)
            if choice == '1':
                print(bcolors.OKCYAN+'\nNEW EXPERIMENT'+bcolors.ENDC)
                endOfExperiment = True
            elif choice == '2':
                print(bcolors.OKCYAN+'Showing graphs.'+bcolors.ENDC)
                autoencoder_visualization(histories, data)
            elif choice == '3':
                savePath = input(bcolors.OKCYAN+'Save model on: '+bcolors.ENDC)
                autoencoder.save(savePath)
                print(bcolors.OKCYAN+'Saved model.'+bcolors.ENDC)
            elif choice == '4':
                print(bcolors.BOLD+bcolors.OKCYAN+'Exiting Program.\n'+bcolors.ENDC)
                endOfExperiment = True
                repeat = False
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

# Execution: python autoencoder.py –d <dataset>
if __name__ == "__main__":
    main()
