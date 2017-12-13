import sys
import os
from PIL import Image #Image for writing
import numpy as np #Processing
import math
import plotly.plotly as py
import plotly
import plotly.graph_objs as go


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
        usecols=(1,2,3,5)
    )

    print("Generating heatmap_" + sys.argv[2] + ".JPEG")

    bucketSize = int(sys.argv[3])
    imageBoundsX = int(sys.argv[4]) * int( sys.argv[3])
    imageBoundsY = int(sys.argv[5]) * int(sys.argv[3])
    outImage = Image.new('RGB',(imageBoundsX, imageBoundsY))
    pixels = outImage.load()

    # X and Y bucket counts
    bucketCountx = int(imageBoundsX/bucketSize)
    bucketCounty = int(imageBoundsY/bucketSize)

    # Empty Array
    bucketsError = np.zeros(shape=(bucketCounty, bucketCountx))
    bucketsCount = np.zeros(shape=(bucketCounty, bucketCountx))
    bucketsMaxMAE = np.zeros(shape=(bucketCounty, bucketCountx))
    bucketsMinMAE = np.ones(shape=(bucketCounty, bucketCountx)) * 100

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

        MAE = abs(s[3] - s[2])

        if(MAE > bucketsMaxMAE[buckety][bucketx]):
            bucketsMaxMAE[buckety][bucketx] = MAE;

        if(MAE < bucketsMinMAE[buckety][bucketx]):
            bucketsMinMAE[buckety][bucketx] = MAE;

        if(MAE > max_mae):
            max_mae = MAE;

        if(MAE < min_mae):
            min_mae = MAE;

        bucketsError[buckety][bucketx] += MAE
        bucketsCount[buckety][bucketx] += 1
        

    print(max_mae)
    print(min_mae)
    

    # Projection slice along Y into XZ space (Conventionally XY)
#    for ly in range(0, bucketCounty):
#        for lx in range(0, bucketCountx):
#            # calc MAE and remap it
#            if(bucketsCount[ly][lx] == 0):
#                bucketColour = (35,35,35)
#            else:
#                if(sys.argv[6] == "AVG"):
#                    MAE = bucketsError[ly][lx] / bucketsCount[ly][lx]
#                elif(sys.argv[6] == "MAX"):
#                    MAE = bucketsMaxMAE[ly][lx]
#                else:
#                    MAE = bucketsMinMAE[ly][lx]
#
#                #mappedMAE = remap(MAE, min_mae, max_mae, 0, 2)
#                mappedMAE = remap(MAE, 0, max_mae, 0, 2)
#            
#                # Lerp between colours
#                if(mappedMAE < 1):
#                    bucketColour = lerp((10,10,255),(10,255,10),mappedMAE)
#                else:
#                    bucketColour = lerp((10,255,10),(255,10,10),(mappedMAE - 1))
#
#            # Colour the bucket
#            for suby in range(0, bucketSize):
#                for subx in range(0, bucketSize):
#                    pixels[lx * bucketSize + subx, (imageBoundsY-1) - (ly * bucketSize + suby)] = convToInts(bucketColour)
#
#    outImage.save("heatmap_" + sys.argv[2] + ".TIFF")

    for ly in range(0, bucketCounty):
        for lx in range(0, bucketCountx):
            # Preproc AVG
            if(bucketsCount[ly][lx] == 0):
                bucketsCount[ly][lx] = 1
            # Preproc MAX
            # Preproc MIN
            if(bucketsMinMAE[ly][lx] == 100):
                bucketsMinMAE[ly][lx] = 0

    if(sys.argv[6] == "AVG"):
        plotData = bucketsError / bucketsCount
        plotly.offline.plot({
            "data": [
                go.Heatmap(
                    z=plotData,
                    colorscale=[
                        [0, 'rgb(35,35,35)'],
                        [1e-20, 'rgb(10,10,255)'],
                        [0.5, 'rgb(10,255,10)'],
                        [0.5, 'rgb(10,255,10)'],
                        [1, 'rgb(255,10,10)']
                    ]),
                ],
            "layout": go.Layout(
                title="Average MAE (m/s) at " + sys.argv[2] + " meters",
                xaxis=dict(title='Binned Distance from Origin in Y'),
                yaxis=dict(title='Binned Distance from Origin in Z')
            )
        },
        filename='heatmap-' + sys.argv[2] + '-avg.html')

        #plotly.offline.plot({"data":[{
        #    'z': plotData,
        #    'type': 'heatmap',
        #    'colorscale': [
        #        #Invalids
        #        [0, 'rgb(35,35,35)'],
        #        #[0.00001, 'rgb(20,20,20)'],
#
        #        [1e-20, 'rgb(10,10,255)'],
        #        [0.5, 'rgb(10,255,10)'],
        #        [0.5, 'rgb(10,255,10)'],
        #        [1, 'rgb(255,10,10)']
        #    ],
        #    'colorbar': {
        #        'tick0': 0,
        #        'dtick': 1
        #    }}],
        #    "layout": Layout(title="Hello"),
        #    filename='heatmap-' + sys.argv[2] + '-avg')

    elif(sys.argv[6] == "MAX"):
        plotData = bucketsMaxMAE
        plotly.offline.plot({
            "data": [
                go.Heatmap(
                    z=plotData,
                    colorscale=[
                        [0, 'rgb(35,35,35)'],
                        [1e-20, 'rgb(10,10,255)'],
                        [0.5, 'rgb(10,255,10)'],
                        [0.5, 'rgb(10,255,10)'],
                        [1, 'rgb(255,10,10)']
                    ]),
                ],
            "layout": go.Layout(
                title="Average MAE (m/s) at " + sys.argv[2] + " meters",
                xaxis=dict(title='Binned Distance from Origin in Y'),
                yaxis=dict(title='Binned Distance from Origin in Z')
            )
        },
        filename='heatmap-' + sys.argv[2] + '-max.html')   
    else:
        plotData = bucketsMinMAE
        plotly.offline.plot({
            "data": [
                go.Heatmap(
                    z=plotData,
                    colorscale=[
                        [0, 'rgb(35,35,35)'],
                        [1e-20, 'rgb(10,10,255)'],
                        [0.5, 'rgb(10,255,10)'],
                        [0.5, 'rgb(10,255,10)'],
                        [1, 'rgb(255,10,10)']
                    ]),
                ],
            "layout": go.Layout(
                title="Average MAE (m/s) at " + sys.argv[2] + " meters",
                xaxis=dict(title='Binned Distance from Origin in Y'),
                yaxis=dict(title='Binned Distance from Origin in Z')
            )
        },
        filename='heatmap-' + sys.argv[2] + '-min.html')


if __name__ == '__main__':
    main()
