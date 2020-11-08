
# import needed packages
import numpy as np
import os
import sys
import struct
from array import array as pyarray

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
def load_mnist(dataset, digits=np.arange(10), numOfImages = -1):
    fname_img = os.path.join(".", dataset)
    imgFile = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", imgFile.read(16))
    img = pyarray("B", imgFile.read())
    imgFile.close()

    if numOfImages == -1:
        numOfImages = size #int(len(ind) * size/100.)
    images = np.zeros((numOfImages, rows, cols), dtype=np.uint8)
    for i in range(numOfImages): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ i*rows*cols : (i+1)*rows*cols ]).reshape((rows, cols))
    return images

# Define class for reading hyperparameters
def read_hyperparameters():
    # Number of convolutional layers
    validInput = False
    while not validInput:
        numOfLayers = input(bcolors.OKCYAN+'Give number of convolutional layers: '+bcolors.ENDC)
        try:
            numOfLayers = int(numOfLayers)
            if numOfLayers > 0:
                validInput = True
            else:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

    numOfFilters = list()
    sizeOfFilters = list()
    for i in range(numOfLayers):
        # Number of filters of ith convolutional layer
        validInput = False
        while not validInput:
            numOfFilters_str = input(bcolors.OKCYAN+'Give number of convolutional filters for layer number '+ str(i+1)+ ': '+bcolors.ENDC)
            try:
                if int(numOfFilters_str) > 0:
                    numOfFilters.append(int(numOfFilters_str))
                    validInput = True
                else:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            except ValueError:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    
        # Size of filters of ith convolutional layer
        validInput = False
        while not validInput:
            sizeOfFilters_str = input(bcolors.OKCYAN+'Give size of convolutional filters for layer number '+str(i+1)+': '+bcolors.ENDC)
            try:
                x, y = sizeOfFilters_str.split()
                if int(x) > 0 and int(y) > 0:
                    sizeOfFilters.append((int(x), int(y)))
                    validInput = True
                else:
                    print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
            except ValueError:
                print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

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
    
    print(numOfLayers)
    print(numOfFilters)
    print(sizeOfFilters)
    print(epochs)
    print(batch_size)
    return numOfLayers, numOfFilters, sizeOfFilters, epochs, batch_size


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
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], ' -d <dataset>'+bcolors.ENDC)
            sys.exit()
        datasetFile = sys.argv[sys.argv.index('-d')+1]

    # Reading dataset
    data = load_mnist(datasetFile)
    
    # Executer experiment
    repeat = True
    while repeat:
        # Reading hyperparameters from user
        numOfLayers, numOfFilters, sizeOfFilters, epochs, batch_size = read_hyperparameters()
        print(bcolors.BOLD+'TESTING'+bcolors.ENDC)

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
            elif choice == '3':
                savePath = input(bcolors.OKCYAN+'Save model on: '+bcolors.ENDC)
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
