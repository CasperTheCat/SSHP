import UniformSampler
import numpy as np
import sys
sys.path.insert(0, "/home/cwa3/SSHPGIT/SSHP/Functions")
import GenerateEven
import UniformSampler
import FastLoadCSV


def autoSample(dataset, filename_out="out.csv"):
    """Automatically resample a given dataset"""
    # Given a dataset, we need to know the maximum x, y and z values.
    arr = FastLoadCSV.iter_loadtxt(dataset ,delimiter=',',skiprows=1)
    xvals=arr[:,0]
    yvals=arr[:,1]
    zvals=arr[:,2]

    min_x=min(xvals)
    max_x=max(xvals)
    min_y=min(yvals)
    max_y=max(yvals)
    min_z=min(zvals)
    max_z=max(zvals)

    # Generate a dataset from above
    res = GenerateEven.np_generate_even(
        (min_x, min_y, min_z),
        (max_x, max_y, max_z)
    )

    # Resample against
    resampled = UniformSampler.resampler_from_np(
        arr,
        res,
        filename_out
    )

    # Append resampled to res to form X,Y,Z,V
    print(res.shape)
    print(res[100])
    print(resampled.shape)
    print(resampled[100])
    
    out = np.concatenate((res,resampled), axis=1)

    np.savetxt(
        filename_out,
        out,
        delimiter=','
    )




if __name__ == '__main__':
    autoSample(sys.argv[1],sys.argv[2])