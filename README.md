# TIGER
Topology-Independent Graph-based Event Reconstruction for particle decays. This repository was used in []. 
The ttbar dataset can be taken from [] and the ttH datasets from []. 

### Data preparation 
The samples need to be transformed into the format used for training. The corresponding code is provided in `dataprep`.

### Training 
After costumizing all paths in the config file training can be done by running `python train.py path/to/config.yml`

