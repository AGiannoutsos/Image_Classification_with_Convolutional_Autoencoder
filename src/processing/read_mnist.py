


def read_mnist(file_path):
    with open(file_path, 'rb') as f: 
        a = f.read(16)
        print(type(a),a)



if __name__ == '__main__':
    print("hello world")
    read_mnist("../../data/train-images-idx3-ubyte")