import sys
import os
from PIL import Image #Image for writing
import pandas as pd
import numpy as np #Processing

def main():
    """ Stitch for other CSVs """
    n = pd.DataFrame({'X': []})
    a = pd.read_csv(sys.argv[1], delimiter=',', skiprows=5,names=['Y', 'Z'])
    b = pd.read_csv(sys.argv[2], delimiter=',', skiprows=5,names=['Z', 'V'])
    b = b.drop('Z', 1)
    c = pd.DataFrame({'VI': []})
    res = pd.concat([n,a,b,c], axis=1)
    res = res.apply(pd.to_numeric, args=('coerce',))
    res = res[res.X != 0]
    res = res[res.Y != 0]
    res = res[res.Z != 0]
    res = res[res.V != 0]
    res['VI'] = 10
    res['X'] = sys.argv[3]

    res = res.dropna(subset = ['X', 'Y', 'Z', 'V'])
    #res.head(100000).to_csv("output.csv", index = False)
    res.to_csv(sys.argv[4], index = False)


if __name__ == '__main__':
    main()
