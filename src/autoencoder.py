
import sys

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

def load_mnist(dataset="training", digits=np.arange(10), path=".", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels

print('hi boys', len(sys.argv), 'arguments!')
print('argument list:', str(sys.argv))

# Reading arguments
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
    print('<-d>:', datasetFile)

# Read hyperparameters from user
validInput = False
while not validInput:
    numOfLayers = input('Give number of convolutional layers: ')
    try:
        numOfLayers = int(numOfLayers)
        validInput = True
    except ValueError:
        print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)

numOfFilters = list()
sizeOfFilters = list()
for i in range(0, numOfLayers):
    print('Layer: ', i)
    validInput = False
    while not validInput:
        numOfFilters_str = input('Give number of convolutional filters for layer number '+ str(i+1)+ ': ')
        try:
            numOfFilters.append(int(numOfFilters_str))
            validInput = True
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)
    validInput = False
    while not validInput:
        sizeOfFilters_str = input('Give size of convolutional filters for layer number '+str(i+1)+': ')
        try:
            x, y = sizeOfFilters_str.split()
            sizeOfFilters.append((int(x), int(y)))
            validInput = True
        except ValueError:
            print(bcolors.FAIL+'Error: invalid input.'+bcolors.ENDC)



