# Gen-AIr

The files are organised according to their respective function within the project structure.

The repository currently supports two workflows:

1. a geometric variational autoencoder, which reconstructs an airfoil profile from its own geometry;
2. a supervised inverse model, which predicts an airfoil profile from aerodynamic parameters.

## Repository Structure

```text
PROJECT_ROOT/
├── main_trainer.py
├── test_output.py
├── train.py
├── predict.py
├── src/
│   ├── Autoencoder_m1.py
│   ├── config.py
│   ├── main_data.py
│   ├── point_extract.py
│   └── run_xfoil.py
└── data/
    ├── historical/
    │   ├── Airfoil_txt/
    │   └── polars_txt/
    └── models/
```
## point_extract.py

The point_extract.py script is responsible for reading the folder containing the airfoil profile curves and extracting the corresponding geometric data.

The path to the dataset location can be modified as required. By uncommenting specific commands, it is also possible to choose between extracting profile points or only PARSEC coefficients.

## main_data.py

The main_data.py file prepares the datasets and dataloaders used by the training scripts.

It supports both:

the original geometric dataset used by the autoencoder;
the inverse dataset built from airfoil profiles and aerodynamic polar data.

The inverse dataset is constructed from:
```
data/historical/Airfoil_txt/
data/historical/polars_txt/
```
Each inverse-model sample has the form:
```
input  = [log10(RE), Ncrit, alpha, cl, cd, cdp, cm, top_xtr, bot_xtr]
output = [100 ordinate y del profilo]
```
where the output is composed of:
```
50 ordinate for the upper surface
50 ordinate for the lower surface
```
## Autoencoder_m1.py

The Autoencoder_m1.py file contains the neural network models used in the project.

It includes:

Autoencoder, used by the original geometric VAE workflow;
AeroToAirfoil, used by the supervised inverse model.

The Autoencoder model performs:
```
airfoil profile -> latent representation -> reconstructed airfoil profile
```

The AeroToAirfoil model performs:
```
[log10(RE), Ncrit, alpha, cl, cd, cdp, cm, top_xtr, bot_xtr] -> [100 y-coordinates of the profile]
```
## main_trainer.py

The main_trainer.py script is used to train the original geometric VAE.

It uses the geometric dataloader from main_data.py and trains the network to reconstruct airfoil profiles from their own coordinates.

## test_output.py

The test_output.py script is used to display the results of the original geometric VAE.

It compares an original airfoil profile with the profile reconstructed by the autoencoder.

## train.py

The train.py script is used to train the supervised inverse model.

The model learns to predict the geometry of an airfoil profile from aerodynamic parameters and polar data.

## predict.py

The predict.py script is used to test the supervised inverse model.

It takes aerodynamic parameters as input, generates the corresponding airfoil profile, and can run XFOIL on the generated profile to evaluate the resulting aerodynamic coefficients.

## run_xfoil.py

The run_xfoil.py file contains utility functions for writing generated airfoil profiles in XFOIL-compatible format and running XFOIL analyses on them.

## Network Weights

The weights of the trained networks are stored in .pth files.

These files must be located inside:
```
data/models/
```