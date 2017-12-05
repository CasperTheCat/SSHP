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
        usecols=(1,2,3,5)
    )

    imageBoundsX = 8192
    imageBoundsY = 2048
    outImage = Image.new('RGBA',(imageBoundsX, imageBoundsY))
    pixels = outImage.load()

    # row 0 is Y
    # row 1 is Z
    # row 2 is V
    # row 3 is VP

    # Projection slice along Y into XZ space (Conventionally XY)
    for row in data:
        if(True or row[1] < 100 and row[1] > 100): # Project range
            # Compute delta
            delta = abs(row[3] - row[2])
            #print(delta)
            normalisedDelta = ((delta / row[2]))
            #print(normalisedDelta)
            mappedDelta = max(min(int(normalisedDelta * 255),255),0)
            # Where does this item go?
            pixX = int(round( (row[0] * 100) ))
            pixY = int(round( (2048 - (row[1] * 100)) ))

            #print("writing")
            #print(mappedDelta)
            #print("to")
            #print(pixX)
            #print(pixY)
            #print("\n\n")

            if(pixX in range(0, imageBoundsX) and
                pixY in range(0, imageBoundsY)):
                # Valid move, Check alpha
                temp = pixels[pixX,pixY]

                if(pixels[pixX,pixY][3] == 0):
                    pixels[pixX,pixY] = (mappedDelta,1,mappedDelta,1)
                else:
                    pixels[pixX,pixY] = (temp[0] + mappedDelta, temp[1] + 1, int((temp[0] + mappedDelta) / (temp[1] + 1)), 1)

    outImage.save("projection_" + sys.argv[2] + ".JPEG")

if __name__ == '__main__':
    main()
