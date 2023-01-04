# ML GBT SETI Project
This is the development repository for the search algorithm using semi-unsupervised technique for Summer Research @UC Berkeley SETI.
***Please note that the repository alone is messy, please follow the menu directory in this readme to browse the code base.*** 

# Original Datasets
The original training dataset is described in our paper. Specific links to download the dataset are the following for TRAINING:
[HIP110750](http://seti.berkeley.edu/opendata/?onLoad=1&target=HIP110750&telescopes=["gbt"]&fileTypes=["hdf5"]&dataTypes=["fine"])
[HIP13402](http://seti.berkeley.edu/opendata/?onLoad=1&target=HIP13402&telescopes=["gbt"]&fileTypes=["hdf5"]&dataTypes=["fine"])
[HIP8497](http://seti.berkeley.edu/opendata/?onLoad=1&target=HIP8497&telescopes=["gbt"]&fileTypes=["hdf5"]&dataTypes=["fine"])
*Note: HIP13402 appears in both the training set AND in the top 8 candidates. However this does NOT mean that we tested or validated on trained data, there are multiple observations of HIP13402 conducted at different times. Thus is not an issue.* 

Preprocessing methods were described in the paper.

# Simulation Code
Please find the simulation code in scripts [here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/test_bench/synthetic_real_dynamic.py)

This requires an installation of `SETIGEN` which can be found here 
```bash
pip install setigen
```

# $\beta$-VAE Model
Please find the model training in the following notebook [here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/test_bench/VAE_NEW_ACCELERATED-BLPC1-8hz-1.ipynb).
To get the trained weights it is located in this repository as well [here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/test_bench/VAE-BLPC1-ENCODER_compressed_512v13-0.h5)

# Random Forest Model
We train the random forest in the following script [here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/test_bench/test_real_full_dynamic_forest.py)

# Benchmarking Model
We benchmarked the results in the following notebook [here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/test_bench/Benchmark_paper_final.ipynb)

# Search Targets
Here is the search list containing all the targets [here](https://github.com/PetchMa/ML_GBT_SETI/blob/4096_pipeline/data_archive/L_band_directory.csv)

# Visualisation
The visualisation notebook is too large to be place in a github repository. It simply provides the plots visualised in the paper and does not include any scientific novelty. 
