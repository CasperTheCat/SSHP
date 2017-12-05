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
        skip_header=5
        #usecols=(0,1,2,4)
    )

    imageBoundsX = 8192
    imageBoundsY = 2048
    outImage = Image.new('RGBA',(imageBoundsX, imageBoundsY))
    pixels = outImage.load()

    # Projection slice along Y into XZ space (Conventionally XY)
    for row in data:
        if(True or row[1] < 100 and row[1] > 100): # Project range
            # Compute delta
            pixX = int(round( (row[1] * 100) ))
            pixY = int(round( (imageBoundsY - (row[2] * 100)) ))

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
        
                red = temp[0]
                green = temp[1]
                blue = temp[2]

                incRed = 25
                incGreen = 50
                incBlue = 75
            
                if((blue + incBlue) > 255):
                    if((green + incGreen) > 255):
                        red += incRed
                    green += incGreen
                blue += incBlue

                pixels[pixX,pixY] = (red,green,blue, 1)

    outImage.save("heatmap_" + sys.argv[1] + ".JPEG")

if __name__ == '__main__':
    main()
