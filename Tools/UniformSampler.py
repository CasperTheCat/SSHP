import numpy as np
import sys
from scipy import spatial as sp
sys.path.insert(0, "/home/cwa3/SSHPGIT/SSHP/Functions")
import NearestNeighbours
import FastLoadCSV


def resampler(sampler_set, target_set, filename_out):
    data = FastLoadCSV.iter_loadtxt(sampler_set ,delimiter=',',skiprows=1)

    points=data[:,0:3]
    vels=data[:,3]
    res = np.zeros(points.shape[0])
    print(points.shape[0])

    tree = sp.KDTree(points)
    for i in range(0,points.shape[0]):
        res[i] = NearestNeighbours.getAverageFromNeighboursFromTree(
                points,
                (points[i][0], points[i][1], points[i][2]),
                vels,
                tree,
                kNearest=4
            )

        if(i % 1000 == 0):
            print(str(i) + '/' + str(points.shape[0]))
        
    
    np.savetxt(
        filename_out,
        res,
        delimiter=','
    )

def resampler_from_np(data, target_arr, filename_out):
    #data = sampler_arr


    points=data[:,0:3]
    vels=data[:,3]

    res = np.zeros((target_arr.shape[0],1))

    print(points.shape[0])

    tree = sp.KDTree(points, leafsize=1000000)

    for i in range(0,target_arr.shape[0]):
        res[i][0] = NearestNeighbours.getAverageFromNeighboursFromTree(
                points,
                (target_arr[i][0], target_arr[i][1], target_arr[i][2]),
                vels,
                tree,
                kNearest=3
            )

        if(i % 1000 == 0):
            print(str(i) + '/' + str(target_arr.shape[0]))
        
    
    #np.savetxt(
    #    filename_out,
    #    res,
    #    delimiter=','
    #)
    return res

if __name__ == '__main__':
    resampler(sys.argv[1], sys.argv[2])


    