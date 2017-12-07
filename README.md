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

## Tools
---

### AppendCSV
Used for adding the predicted velocity to the CSV

#### Usage
```bash
python AppendCSV.py <DataCSV> <PredictionCSV>
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

### MAE
This is the python verson of MAE. It is slower but doesn't suffer the rounding error of the C++ version.

#### Usage
```bash
python MAE.py <in.csv>
```

### Stitch
Stitch actually does the append operation, taking two full CSVs and creating a CSV with the rows of both. 

#### Usage
```bash
python Stitch.py <in1.csv> <in2.csv> <out.csv>
```

## Visualisation
---

### Generate Heatmap
This generates a heatmap using bins.

#### Usage
```bash
python GenerateHeatmap.py <in.csv> <heatmapName> <bucketSize> <nBucketsX> <nBucketsY> <mode>
```
where valid modes are MAX, AVG and MIN
