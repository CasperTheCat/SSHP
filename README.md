# SSH
Scripts and Code for SSHP

## C++
---

### MAE.cxx
This is a C++ file for calculating the Mean Absolute Error for a given file.

This requires the actual velocity to be in the 4th row and the predicted velocity to be in the 6th.

This implementation has rounding errors.

#### Compilation
```bash
c++ -std=c++11 MAE.cxx -o MAE.exe
```

#### Usage
```bash
./MAE.exe <in.csv>
```

## Network Documentation
---

### Morte.py
```bash
python Morte.py <train.csv> <validation.csv> <predict1.csv> <predict10.csv> <predict20.csv> <predict30.csv> <predict40.csv> <predict50.csv> <predict60.csv> <predict66.csv>
```
*This file is outdated compared to Revenant.py*

This file is a lower level tensorflow feedforward network using relu activation.

### Revenant.py
```bash
python revenant.py <train.csv> <validation.csv> [model_directory]
```

*Please note that the model directory is local*

This file was originally one half of the Morte.py file. This file generates checkpoints of the network and saves them to disk for use in Renegade.py

### Renegade.py
```bash
python renegade.py <10x15_model_directory> <10x20_model_directory> <15x20_model_directory>
<predict1.csv> <predict10.csv> <predict20.csv> <predict30.csv> <predict40.csv> <predict50.csv> <predict60.csv> <predict66.csv>
```

This file was the other half of the Morte.py file. It takes the predicted sets from Revenant.py and predicts using them.

### Nouveau.py
```bash
python nouveau.py <train.csv> <validation.csv> <predict10.csv> <predict20.csv> <predict30.csv> <predict40.csv> <predict50.csv> <predict60.csv> <predict66.csv>
```

*This file is deprecated*

This file uses the DNN Regressor to train and predict a network. This file was deprecated as I was unable to get the export and import for the canned estimators working.


## Tools
---

### Align
This file adjusts the predicted values by the mean error

#### Usage
```bash
python Align.py <DataAndPredictionCSV>
```

### AppendCSV
Used for adding the predicted velocity to the CSV

#### Usage
```bash
python AppendCSV.py <DataCSV> <PredictionCSV>
```

### AutoSampler
This automatic resamples the dataset into uniform spaced area in 3D space

Be careful of the memory usage during creation and performance impact when training on a dense resampled dataset

#### Usage
```bash
python AutoSampler.py <inputCSV> <outputCSV>
```

### CombineCSV and CombineCSVNX
Both of these deal with combining the CFD data into one CSV.

CombineCSV takes two CSVs and combines them, adding a VI column.

CombineCSVNX takes two CSVs, combining them in a different manner. This is for use on CSVs where the XYZ columns are incorrect.

#### Usage
```bash
python CombineCSV.py <xy.csv> <velz.csv> <out.csv>
```
```bash
python CombineCSVNX.py <xy.csv> <velz.csv> <XValue> <out.csv>
```

### HalfCSV
Cut the CSV in half and output both halves

#### Usage
```bash
python HalfCSV.py <in1.csv> <out1.csv> <out2.csv>
```

### MAE
This is the python verson of MAE. It is slower but doesn't suffer the rounding error of the C++ version.

#### Usage
```bash
python MAE.py <in.csv>
```

### PackSimDataToImage
This is not actually a tool. It instead is a helper that turns the simulation into an image format


### Stitch
Stitch actually does the append operation, taking two full CSVs and creating a CSV with the rows of both. 

#### Usage
```bash
python Stitch.py <in1.csv> <in2.csv> <out.csv>
```

### UniformSampler
This creates a uniformly sampled dataset. It is used by autosampler.

This file requires an updated sampler(...) in order to work in standalone.


## Visualisation
---

### Generate Heatmap
This generates a heatmap using bins.

#### Usage
```bash
python GenerateHeatmap.py <in.csv> <heatmapName> <bucketSize> <nBucketsX> <nBucketsY> <mode>
```
where valid modes are MAX, AVG and MIN

### GenerateCountMap
This generates a map of number of samples using bins.

#### Usage
```bash
python GenerateCountMap.py <in.csv> <heatmapName> <bucketSize> <nBucketsX> <nBucketsY> <mode>
```
where valid modes are MAX, AVG and MIN