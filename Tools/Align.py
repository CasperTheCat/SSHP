import sys
import os
from PIL import Image #Image for writing
import numpy as np #Processing


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
    """Class"""
    fileToLoad=sys.argv[1]
    data = iter_loadtxt(
        fileToLoad,
        delimiter=',',
        skiprows=1
    )

    os.nice(20)
    # row 0 is Y
    # row 1 is Z
    # row 2 is V
    # row 3 is VP
    summ = 0
    count = 0
    print("Processing")
    for row in data:
        summ += (row[3] - row[6])
        count += 1


    avgError = summ /count
    print("Adjusting error by average of %f" % avgError)

    # Adjust the second sys.argv? or just edit
    for row in data:
        row[6] += (avgError * 0.5)

    np.savetxt(sys.argv[1], data, delimiter=',', header="X,Y,Z,V,VI,Theta,VP")




if __name__ == '__main__':
    main()