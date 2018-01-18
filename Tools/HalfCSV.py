import sys
import os
from PIL import Image #Image for writing
import pandas as pd
import numpy as np #Processing

def main():
    """Class"""
    a = pd.read_csv(sys.argv[1], delimiter=',', skiprows=5,names=['X', 'Y','Z','V','VI'])
    a = a.sample(frac=1).reset_index(drop=True)

    total_rows = a.shape[0]
    print(total_rows)

    b = a.iloc[:int(total_rows/2)]
    c = a.iloc[int(total_rows/2):]
    

    #b = pd.read_csv(sys.argv[2], delimiter=',', skiprows=5,names=['VP'])
    #res = pd.concat([a,b], axis=1)
    #res = res.apply(pd.to_numeric, args=('coerce',))
    #res['VI'] = 10

    #res = res.dropna(subset = ['X', 'Y', 'Z', 'V', 'VP'])
    #res.head(100000).to_csv("outputwithres.csv", index = False)
    b.to_csv(sys.argv[2], index = False)
    c.to_csv(sys.argv[3], index = False)


if __name__ == '__main__':
    main()

