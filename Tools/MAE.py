import sys
import os
from PIL import Image #Image for writing
import numpy as np #Processing

def main():
    """Class"""
    fileToLoad=sys.argv[1]
    data = np.genfromtxt(
        fileToLoad,
        delimiter=',',
        skip_header=1,
        usecols=(3,5)
    )

    os.nice(20)
    # row 0 is Y
    # row 1 is Z
    # row 2 is V
    # row 3 is VP
    summ = 0
    count = 0
    
    for row in data:
        summ += abs(row[0] - row[1])
        count += 1

    print(summ)
    print(count)
    print(summ / count)


if __name__ == '__main__':
    main()
