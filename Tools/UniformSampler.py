import numpy as np
sys.path.insert(0, "/home/cwa3/SSHPGIT/SSHP/Functions")
import NearestNeighbours

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def main():
    data = iter_loadtxt(VAL,delimiter=',',skiprows=1)

    points=data[:,0:3]
    vels=data[:,3]

    

    