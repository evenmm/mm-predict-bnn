# mm-predict-bnn
This repository contains the code to reproduce the results in the article "Prediction of cancer dynamics under treatment using Bayesian neural networks: A simulated study".

The code was run with Python Python 3.8.16 using the packages listed in requirements.txt. 

Bayesian inference to produce binaries and pickles were carried out using the med-biostat2 server of Oslo centre for biostatistics and epidemiology.
Utility scripts *run_jobs.sh* and *screenkill.sh* were used to parallellize the inference using screen instances. 

Boxplots, and plots of individual patients, were not generated on the server but produced locally using *plot_cluster_results_locally.py*.
