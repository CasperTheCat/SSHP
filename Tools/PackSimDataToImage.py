import sys
import os
from PIL import Image #Image for writing
import numpy as np #Processing

def main():
    """Class"""
    firstFileToLoad=sys.argv[1]
    data = np.genfromtxt(
        firstFileToLoad,
        delimiter=',',
        skip_header=5
        #usecols=(0,1,2,4)
    )

    secondFileToLoad=sys.argv[2]
    datas = np.genfromtxt(
        secondFileToLoad,
        delimiter=",",
        skip_header=5
    )

    imageBoundsX = 8192
    imageBoundsY = 8192
    outImage = Image.new('F',(imageBoundsX, imageBoundsY))
    pixels = outImage.load()

    i = 0

    # Projection slice along Y into XZ space (Conventionally XY)
    for row in data:
        if(i < 1000000):
            # Valid move, Check alpha
            #print("\nX")
            #print(row[0])
            #print("Y")
            #print(row[1])
            #print("Z")
            #print(datas[i][0])
            #print(datas[i][1])


            pixels[int(i / imageBoundsX),int(i % imageBoundsY)] = row[0]
            pixels[int((i+1) / imageBoundsX),int((i+1) % imageBoundsY)] = row[1]
            pixels[int((i+2) / imageBoundsX),int((i+2) % imageBoundsY)] = datas[i][0]
            pixels[int((i+3) / imageBoundsX),int((i+3) % imageBoundsY)] = datas[i][1]
            i+=1

    outImage.save("packe_" + "new" + ".tiff")

    for row in data:
        if(i < 1000000):
            



            i+=1

if __name__ == '__main__':
    main()
