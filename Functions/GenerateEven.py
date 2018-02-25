import sys
import numpy as np

def csv_generate_even(fout,min,max,distance=0.05):
    """
    Generate an evenly distributed CSV
    """
    x_min, y_min, z_min = min
    x_max, y_max, z_max = max

    with open(fout, "w+") as fl:
        for i in range(x_min, x_max):
            for id in range(0, int(1/distance)):
                for j in range(y_min, y_max):
                    for jd in range(0, int(1/distance)):
                        for k in range(z_min, z_max):
                            for kd in range(0, int(1/distance)):
                                print(i + id * distance)
                                fl.write(
                                    str(i + id * distance) +
                                    "," +
                                    str(j + jd * distance) +
                                    "," +
                                    str(k + kd * distance) +
                                    ",0,10,0\n"
                                    )
                    # generate stepping

    fl.close()
                
def np_generate_even(mins,maxs,distance=0.1):
    """
    Generate an evenly distributed numpy array
    """
    x_min, y_min, z_min = mins
    x_max, y_max, z_max = maxs

    print(mins)
    print(maxs)

    spaceMul = int(1/distance)

    xDelta = int(max((x_max - x_min), 1))
    yDelta = int(max((y_max - y_min), 1))
    zDelta = int(max((z_max - z_min), 1))

    print(xDelta)
    print(yDelta)
    print(zDelta)
    print(spaceMul)

    rawIndex = 0
    if(x_min == x_max):
        res = np.zeros(
            (
                (
                    yDelta * spaceMul *
                    zDelta * spaceMul
                ),
                3
            )
        )

        for j in range(0, yDelta):
            for jd in range(0, spaceMul):
                for k in range(0, zDelta):
                    for kd in range(0, spaceMul):
                        res[rawIndex][0] = x_min
                        res[rawIndex][1] = j + jd * distance + y_min
                        res[rawIndex][2] = k + kd * distance + z_min
                        rawIndex += 1
    else:
        res = np.zeros(
            (
                (
                    xDelta * spaceMul *
                    yDelta * spaceMul *
                    zDelta * spaceMul
                ),
                3
            )
        )

        for i in range(0, xDelta):
            for id in range(0, spaceMul):
                for j in range(0, yDelta):
                    for jd in range(0, spaceMul):
                        for k in range(0, zDelta):
                            for kd in range(0, spaceMul):
                                res[rawIndex][0] = i + id * distance
                                res[rawIndex][1] = j + jd * distance
                                res[rawIndex][2] = k + kd * distance
                                rawIndex += 1

    print(res[100])
    return res



if __name__ == '__main__':
    min = (0,0,0)
    max = (100,100,100)
    csv_generate_even(sys.argv[1], min, max)
