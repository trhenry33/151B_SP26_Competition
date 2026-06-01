# CSE 151B Competition

## GPU
We ran our work on the A30 GPU in the DSMLP Cluster. It should take around 12 hours to generate all results for the private set.

## Setup
To setup run final_setup.py located in the the setup directory. If this does not work, look inside final_setup.py to see the manual installs and uninstalls of numpy and vllm.

## Weights
There are no weights to download.

## How to Run
Run material is all in run.py, just run that on an a30 GPU or one or similar calliber, and then check run_inference_final.csv for results.
If you wish to only run 200, change the run_end_idx, and if you wish to choose a different data file, change the DATA_PATH.








