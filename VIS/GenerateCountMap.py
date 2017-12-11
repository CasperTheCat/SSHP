import sys
import os
from PIL import Image #Image for writing
import numpy as np #Processing
import math

def remap(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)

def lerp(a,b,c):
    return (
        a[0] * (1-c) + b[0] * c,
        a[1] * (1-c) + b[1] * c,
        a[2] * (1-c) + b[2] * c
    )

def convToInts(a):
    return (
        int(round(a[0])),
        int(round(a[1])),
        int(round(a[2]))
    )

def main():
    """Class"""

    if(sys.argv[6] != "MAX" and sys.argv[6] != "AVG" and sys.argv[6] != "MIN"):
        print("only MAX and AVG modes")
        sys.exit()

    fileToLoad=sys.argv[1]
    data = np.genfromtxt(
        fileToLoad,
        delimiter=',',
        skip_header=1,
        usecols=(1,2)
    )

    print("Generating heatmap_" + sys.argv[2] + "." + sys.argv[7])

    bucketSize = int(sys.argv[3])
    imageBoundsX = int(sys.argv[4]) * int( sys.argv[3])
    imageBoundsY = int(sys.argv[5]) * int(sys.argv[3])
    outImage = Image.new('RGB',(imageBoundsX, imageBoundsY))
    pixels = outImage.load()

    # X and Y bucket counts
    bucketCountx = int(imageBoundsX/bucketSize)
    bucketCounty = int(imageBoundsY/bucketSize)

    # Empty Array
    bucketsCount = np.zeros(shape=(bucketCounty, bucketCountx))

    # Remapping
    yvals=data[:,0]
    zvals=data[:,1]
    
    # Y Values
    min_x=min(yvals)
    max_x=max(yvals)

    # Z Values
    min_y=min(zvals)
    max_y=max(zvals)

    # Forward Declaration of MAE Max and Min = 0
    max_mae = 0
    min_mae = 100 # Change if needs be

    # Forward Declaration of 

    for s in data:
        # Which bin?
        mappedx = remap(s[0], min_x, max_x, 0, imageBoundsX - 1)
        mappedy = remap(s[1], min_y, max_y, 0, imageBoundsY - 1)

        bucketx = int(math.floor( mappedx / bucketSize))
        buckety = int(math.floor( mappedy / bucketSize))

        bucketsCount[buckety][bucketx] += 1
   
 
    for ly in range(0, bucketCounty):
        for lx in range(0, bucketCountx):
            if(bucketsCount[ly][lx] > max_mae):
                max_mae = bucketsCount[ly][lx]
        

    print(max_mae)
    print(min_mae)
    

    # Projection slice along Y into XZ space (Conventionally XY)
    for ly in range(0, bucketCounty):
        for lx in range(0, bucketCountx):
            # calc MAE and remap it
            if(bucketsCount[ly][lx] == 0):
                bucketColour = (35,35,35)
            else:
                MAE = bucketsCount[ly][lx]
                mappedMAE = remap(MAE, 0, max_mae, 0, 2)
            
                # Lerp between colours
                if(mappedMAE < 1):
                    bucketColour = lerp((10,10,255),(10,255,10),mappedMAE)
                else:
                    bucketColour = lerp((10,255,10),(255,10,10),(mappedMAE - 1))

            # Colour the bucket
            for suby in range(0, bucketSize):
                for subx in range(0, bucketSize):
                    pixels[lx * bucketSize + subx, (imageBoundsY-1) - (ly * bucketSize + suby)] = convToInts(bucketColour)

    outImage.save("heatmap_" + sys.argv[2] + "." + sys.argv[7])

if __name__ == '__main__':
    main()
