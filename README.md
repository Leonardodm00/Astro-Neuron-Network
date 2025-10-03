
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


## TO DO
- Write down the C++ friendly code.
- Simulate some instaces of different degrees of astrocytic network's influence.
