###  DATASET 

  - dataset without protein data：0507ligand_pro_15cutways.txt  
    The first column is an Smiles sequence of the intact molecule, and the second column is an Smiles sequence of the cropped molecule
  - dataset with protein data：0525ligand_pro_15cutways.txt  
    The first column is an Smiles sequence of the intact molecule, the second column is an Smiles sequence of the clipped molecule, and the third column is a protein sequence. 

### Train and test

```
// run the code, each model code is in the corresponding folder
run LSTM_molecule0507.py
```
### Outputs
Take the * `./LSTM` folder as an example:
* `./r1` ----r1: Experimental results
* `./r1/parameters.txt` : input hyperparameters and loss output
* `./r1/evaluate.txt`: some indicators of prediction results
* `./r1/encoder.pth/decoder.pth:` contains the trained model
* `./LSTMresults` : Model prediction results

