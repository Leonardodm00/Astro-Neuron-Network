
# MAIN CODES
## Activity
- *Main_code.py*: Primary interface to simulate the ***in-silico*** culture (biophysical)
- *ANS_fun.py*: Gathers all the functions employed in *Main_code.py*. Cleared from not used options to construct synapses and with added asyncronous NT release. (biophysical)
- *Main_code_pheno.py*: Phenomenological version of the *Main_code*
- *ANS_fun_pheno.py*; Phenomenological version of the *ANS_fun*
## Growing
- *Culture_growth.py* : Primary interface to simulate the growth the ***in-silico*** culture
- *Culture_growth_class.py*: Main class called in *Culture_growth.py*
- *Growing_fun.py* : Gathers all the function needed to support *Culture_growth_class.py*
*Culture_growth_class.py* and *Growing_fun.py* have the parallel version that explits cpu-based multiprocessing.
*NOT PARALLELIZED VERSIONS ARE DEPRECATED AND NOT KEPT UPDATED*

## HPC
The scripts are used to be run in a HPC.
- Main_code_notebook.ipyn: Main interface to run the code
- ASN_fun_BD_cpp : Main functions repository to run the code in a HPC 


## TO DO
Finish to make the data processing in the wrapper
finish to prepare the functions for the HPC.

