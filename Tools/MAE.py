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
        usecols=(3,6)
    )

    os.nice(20)
    # row 0 is Y
    # row 1 is Z
    # row 2 is V
    # row 3 is VP
    summ = 0
    count = 0

    partp = data[1000000:1050000]
 
    cc = np.mean(np.corrcoef(partp))
    #print("Correlation: %g" % cc)
    print(cc)

    avgError = 0
    di = 0
    for row in data:
        di += 1
        summ += abs(row[0] - row[1])
        avgError += (row[0] - row[1])
        count += 1
        if(di == 1):
            print("Stat %d" % di)
            print(row[0])
            print(row[1])
            print(row[0] - row[1])
            print(abs(row[0] - row[1]))

    avgError /= count
    print(summ)
    print(count)
    print(summ / count)

    print("\n Attempting to correct by average error of %f" % avgError)
    summ = 0
    for row in data:
        summ += abs(row[0] - (row[1] + avgError))
        
    print(summ / count)
    

if __name__ == '__main__':
    main()
