import numpy
import math
import sys
# Reads in a CSV dataset with y,z columns, and resamples the dataset so that samples
# are spread approximately uniformly across the 3D space. x values are assumed to be already
# approimately uniform.

# Delete a row from a numpy array. Return a modified copy of the array.
def delete(arr,row_index):
  return numpy.delete(arr,(row_index),axis=0)


# Resample function, the main function of this script
def resample(infilename,outfilename,num_bins_y,num_bins_z,max_bin_count):
  # Print parameters
  print("infilename:   \t",infilename)
  print("outfilename:  \t",outfilename)
  print("num_bins_y:   \t",num_bins_y)
  print("num_bins_z:   \t",num_bins_z)
  print("max_bin_count:\t",max_bin_count)
  # Parse in the CSV file
  hdr = numpy.genfromtxt(infilename, delimiter=',',max_rows=1,dtype=str).tolist()
  arr = numpy.genfromtxt(infilename, delimiter=',',skip_header=True)
  print("hdr:    \t",hdr)
  print("arr.shape:    \t",arr.shape)
  # Initialise a 2D y/z array for counters
  counts = numpy.zeros((num_bins_y,num_bins_z),dtype=int)
  # Get the max/min y/z values from the data
  y_column_index=hdr.index('Y')
  z_column_index=hdr.index('Z')
  yvals=arr[:,y_column_index]
  zvals=arr[:,z_column_index]
  min_y=min(yvals)
  max_y=max(yvals)
  min_z=min(zvals)
  max_z=max(zvals)
  print("min_y:   \t",min_y)
  print("max_y:   \t",max_y)
  print("min_z:   \t",min_z)
  print("max_z:   \t",max_z)
  # Shuffle the rows of the data array
  numpy.random.shuffle(arr)
  # Open the output file
  outfile = open(outfilename,"w")
  outfile.write(','.join(hdr)+'\n')
  # Iterate over the samples in the data array,
  # working our which bin each sample falls into,
  # and keep the sample only if the max bin count
  # is not exceeded
  y_bin_size=(max_y-min_y)/num_bins_y
  z_bin_size=(max_z-min_z)/num_bins_z
  print("y_bin_size:   \t",y_bin_size)
  print("z_bin_size:   \t",z_bin_size)
  for sample in arr:
    y = sample[y_column_index]
    z = sample[z_column_index]
    ybin = int(math.floor((y-min_y)/y_bin_size))
    zbin = int(math.floor((z-min_z)/z_bin_size))
    if ybin==num_bins_y:
      ybin=num_bins_y-1
    if zbin==num_bins_z:
      zbin=num_bins_z-1
    if counts[ybin][zbin]<max_bin_count:
      outfile.write(','.join(map(str,sample.tolist()))+'\n')
      counts[ybin][zbin]=counts[ybin][zbin]+1
  # Close the output file
  outfile.close()


# Run the script
resample(sys.argv[1],"dataset-resampled.csv",200,200,50)
