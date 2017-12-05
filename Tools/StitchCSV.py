import sys
import os
from PIL import Image #Image for writing
import pandas as pd
import numpy as np #Processing

def main():
    """Class"""
    a = pd.read_csv(sys.argv[1], delimiter=',', skiprows=5,names=['X', 'Y', 'Z', 'V', 'VI'])
    b = pd.read_csv(sys.argv[2], delimiter=',', skiprows=5,names=['X', 'Y', 'Z', 'V', 'VI'])
    res = pd.concat([a,b])
    #res = res.apply(pd.to_numeric, args=('coerce',))

    #res = res.dropna(subset = ['X', 'Y', 'Z', 'V'])
    #res.head(100000).to_csv("output.csv", index = False)
    res.to_csv(sys.argv[3], index = False)


if __name__ == '__main__':
    main()

