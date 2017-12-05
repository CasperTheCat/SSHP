import sys
import os
from PIL import Image #Image for writing
import pandas as pd
import numpy as np #Processing

def main():
    """Class"""
    a = pd.read_csv(sys.argv[1], delimiter=',', skiprows=5,names=['X', 'Y','Z','V','VI'])
    b = pd.read_csv(sys.argv[2], delimiter=',', skiprows=5,names=['VP'])
    res = pd.concat([a,b], axis=1)
    #res = res.apply(pd.to_numeric, args=('coerce',))
    #res['VI'] = 10

    res = res.dropna(subset = ['X', 'Y', 'Z', 'V', 'VP'])
    #res.head(100000).to_csv("outputwithres.csv", index = False)
    res.to_csv(sys.argv[3], index = False)


if __name__ == '__main__':
    main()

