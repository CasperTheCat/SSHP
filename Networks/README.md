# Network Documentation

## Morte.py
```python Morte.py <train.csv> <validation.csv> <predict1.csv> <predict10.csv> <predict20.csv> <predict30.csv> <predict40.csv> <predict50.csv> <predict60.csv> <predict66.csv>```

This file is a lower level tensorflow feedforward network using relu activation

## Revenant.py
```python revenant.py <train.csv> <validation.csv> [model_directory]```

*Please note that the model directory is local*

This file was originally one half of the Morte.py file. This file generates checkpoints of the network and saves them to disk for use in Renegade.py

## Renegade.py
```
python renegade.py <10x15_model_directory> <10x20_model_directory> <15x20_model_directory>
<predict1.csv> <predict10.csv> <predict20.csv> <predict30.csv> <predict40.csv> <predict50.csv> <predict60.csv> <predict66.csv>
```

This file was the other half of the Morte.py file. It takes the predicted sets from Revenant.py and predicts using them.

