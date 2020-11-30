
# import needed packages
import numpy as np
import os
import sys
import struct
import json
from array import array as pyarray
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.utils import to_categorical, normalize
from keras.models import load_model
# inport our files
from model import get_Classifier, train_Classifier
from visualization import classifier_prediction_visualization_window, classifier_loss_visualization_window, classifier_prediction_visualization

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
        # print(data)
        # print(data.shape)
        # print(data[0])
        # file = open(fname, 'rb')
        # magic_nr, size, rows, cols = struct.unpack(">IIII", file.read(16))
        # img = pyarray("B", file.read())
        # file.close()

        # if numOfElements == -1:
        #     numOfElements = size #int(len(ind) * size/100.)
        # images = np.zeros((numOfElements, rows, cols), dtype=np.uint8)
        # for i in range(numOfElements): #int(len(ind) * size/100.)):
        #     images[i] = np.array(img[ i*rows*cols : (i+1)*rows*cols ]).reshape((rows, cols))
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
    
    # Number of convolutional layers on Dense
    validInput = False
    while not validInput:
        numOfLayers = input(bcolors.OKCYAN+'Give number of layers on dense: '+bcolors.ENDC)
        try:
            numOfLayers = int(numOfLayers)
            if numOfLayers > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    
    # Layers of Dense
    model_info['dense_layers'] = list();
    for i in range(numOfLayers):
        # Type of ith convolutional layer
        validInput = False
        while not validInput:
            layer_type = input(bcolors.OKCYAN+'Choose type of layer number '+ str(i+1)+ ' (drop/batchNorm/dense): '+bcolors.ENDC)
            if layer_type == 'drop' or layer_type == 'batchNorm' or layer_type == 'dense':
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

        if layer_type == 'drop':
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
            model_info['dense_layers'].append([layer_type, numOfFilters, (x, y)])

        elif layer_type == 'dense':
            # Size of filters of ith convolutional layer
            validInput = False
            while not validInput:
                density = input(bcolors.OKCYAN+'Give density of the layer: '+bcolors.ENDC)
                try:
                    density = int(density)
                    if density > 0:
                        validInput = True
                    else:
                        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                except ValueError:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            model_info['dense_layers'].append([layer_type, density])

        else: # layer_type == 'batchNorm'
            model_info['dense_layers'].append([layer_type])

    # Number of dense epochs
    validInput = False
    while not validInput:
        dense_only_train_epochs = input(bcolors.OKCYAN+'Give number of dense train epochs: '+bcolors.ENDC)
        try:
            dense_only_train_epochs = int(dense_only_train_epochs)
            if dense_only_train_epochs > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    model_info['dense_only_train_epochs'] = dense_only_train_epochs

    # Number of full model epochs
    validInput = False
    while not validInput:
        full_train_epochs = input(bcolors.OKCYAN+'Give number of total epochs: '+bcolors.ENDC)
        try:
            full_train_epochs = int(full_train_epochs)
            if full_train_epochs > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    model_info['full_train_epochs'] = full_train_epochs
    
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

    model_info['optimizer'] = ["adam", 0.001]

    print(model_info)
    return model_info


# Main Function
def main():
    # print('argument list:', str(sys.argv))
    model_info = {}

    # Reading inline arguments
    # <-d> argument
    if '-d' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-d>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-d') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
            sys.exit()
        datasetFile = sys.argv[sys.argv.index('-d')+1]
    # <-dl> argument
    if '-dl' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-dl>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-dl') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
            sys.exit()
        dlabelsFile = sys.argv[sys.argv.index('-dl')+1]
    # <-t> argument
    if '-t' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-t>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-t') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
            sys.exit()
        testsetFile = sys.argv[sys.argv.index('-t')+1]
    # <-tl> argument
    if '-tl' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-tl>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-tl') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
            sys.exit()
        tlabelsFile = sys.argv[sys.argv.index('-tl')+1]
    # <-model> argument
    if '-model' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-model>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-model') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>'+bcolors.ENDC)
            sys.exit()
        encoder = sys.argv[sys.argv.index('-model')+1]

    # Reading training and test sets
    if not os.path.isfile(datasetFile):
        print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
        sys.exit()
    train_X = normalize(load_mnist(datasetFile, type='data'))
    if not os.path.isfile(dlabelsFile):
        print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
        sys.exit()
    train_Y = to_categorical(load_mnist(dlabelsFile, type='labels'))
    if not os.path.isfile(testsetFile):
        print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
        sys.exit()
    test_X = normalize(load_mnist(testsetFile, type='data'))
    if not os.path.isfile(tlabelsFile):
        print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
        sys.exit()
    test_Y = to_categorical(load_mnist(tlabelsFile, type='labels'))

    
    # Executer experiment
    histories = list()
    repeat = True
    while repeat:
        # Reading hyperparameters from user
        model_info = read_hyperparameters()
        model_info["encoder_layers"] = encoder

        classifier = get_Classifier(model_info, train_X.shape[1:], 10)
        
        # Run Experiment!
        print(bcolors.BOLD+'\nTRAINING'+bcolors.ENDC)
        print(bcolors.BOLD+'----------------------------------------------------'+bcolors.ENDC)
        print(train_X.shape)
        histories.append(train_Classifier(classifier, model_info, train_X, train_Y, test_X, test_Y))

        # Check what user wants to do next
        endOfExperiment = False
        while not endOfExperiment:
            print(bcolors.BOLD+'\n----------------------------------------------------'+bcolors.ENDC+bcolors.OKCYAN)
            print('1. Repeat experiment with diferent hyperparameters.')
            print('2. Show graphs of error.')
            print('3. classification of images in test set.')
            print('4. Exit Program.')
            choice = input('Choose something from above: '+bcolors.ENDC)
            if choice == '1':
                print(bcolors.OKCYAN+'\nNEW EXPERIMENT'+bcolors.ENDC)
                endOfExperiment = True
            elif choice == '2':
                print(bcolors.OKCYAN+'Showing graphs.'+bcolors.ENDC)
                classifier_loss_visualization_window(histories)
            elif choice == '3':
                # Ask if user wants to test a saved model
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
                        if os.path.exists(model_info):
                            validInput = True
                        else:
                            print(bcolors.FAIL+'Error: invalid path.'+bcolors.ENDC)
                    print(bcolors.OKCYAN+'Images classification.'+bcolors.ENDC)
                    classifier_prediction_visualization_window(load_model(model_info), test_X, test_Y)
                else:
                    if len(histories) > 1:
                        validInput = False
                        while not validInput:
                            modelNum = input(bcolors.OKCYAN+'Which model you want to use? (choices: 1-'+str(len(histories))+') '+bcolors.ENDC)
                            try:
                                modelNum = int(modelNum)-1
                                if modelNum >= 0 and modelNum < len(histories):
                                    validInput = True
                                else:
                                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                            except ValueError:
                                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
                    else:
                        modelNum = 0
                    print(bcolors.OKCYAN+'Images classification.'+bcolors.ENDC)
                    classifier_prediction_visualization_window(histories[modelNum].model, test_X, test_Y)
            elif choice == '4':
                print(bcolors.BOLD+bcolors.OKCYAN+'Exiting Program.\n'+bcolors.ENDC)
                endOfExperiment = True
                repeat = False
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

# Execution: python classification.py –d <training_set> –dl <training_labels> -t <test_set> -tl <test_labels> -model <autoencoder_h5>
if __name__ == "__main__":
    main()
