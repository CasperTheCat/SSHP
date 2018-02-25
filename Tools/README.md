## Align
This file adjusts the predicted values by the mean error

### Usage
```bash
python Align.py <DataAndPredictionCSV>
```

## AppendCSV
Used for adding the predicted velocity to the CSV

### Usage
```bash
python AppendCSV.py <DataCSV> <PredictionCSV>
```

## AutoSampler
This automatic resamples the dataset into uniform spaced area in 3D space

Be careful of the memory usage during creation and performance impact when training on a dense resampled dataset

### Usage
```bash
python AutoSampler.py <inputCSV> <outputCSV>
```

## CombineCSV and CombineCSVNX
Both of these deal with combining the CFD data into one CSV.

CombineCSV takes two CSVs and combines them, adding a VI column.

CombineCSVNX takes two CSVs, combining them in a different manner. This is for use on CSVs where the XYZ columns are incorrect.

### Usage
```bash
python CombineCSV.py <xy.csv> <velz.csv> <out.csv>
```
```bash
python CombineCSVNX.py <xy.csv> <velz.csv> <XValue> <out.csv>
```

## HalfCSV
Cut the CSV in half and output both halves

### Usage
```bash
python HalfCSV.py <in1.csv> <out1.csv> <out2.csv>
```

## MAE
This is the python verson of MAE. It is slower but doesn't suffer the rounding error of the C++ version.

### Usage
```bash
python MAE.py <in.csv>
```

## PackSimDataToImage
This is not actually a tool. It instead is a helper that turns the simulation into an image format


## Stitch
Stitch actually does the append operation, taking two full CSVs and creating a CSV with the rows of both. 

### Usage
```bash
python Stitch.py <in1.csv> <in2.csv> <out.csv>
```

## UniformSampler
This creates a uniformly sampled dataset. It is used by autosampler.

This file requires an updated sampler(...) in order to work in standalone.