# Inverse Reinforcement Learning in Contextual MDPs

This repository contains the code files that we used in our work to construct two environments and test three learning methods.
The environments in this repository:
1. Driving simulator
2. Dynamic treatment regime

## To run the simulations clone the repository and run the following files:
### Dynamic treatment regime:
#### Ellipsoid method:
* Dynamic treatment regime/Linear/Ellipsoid/ellipsoid_medical.py  
#### ES with the 1st loss:
* Dynamic treatment regime/Linear/Blackbox_loss1/bb_medical.py  
#### ES with the 2nd loss:
* Dynamic treatment regime/Linear/Blackbox_loss2/bbl2_medical.py  
### Driving simulator:
#### Ellipsoid method:
* Driving simulation/Linear/Ellipsoid/ellipsoid_driving.py  
#### ES with the 1st loss:
* Driving simulation/Linear/Blackbox_loss1/bb_driving.py  
#### ES with the 2nd loss:
* Driving simulation/Linear/Blackbox_loss2/bbl2_driving.py  
#### ES with the 2nd loss on the non-linear model:
* Driving simulation/non_linear/Blackbox_loss2/bbl2_driving.py

## Plot the results:
* use the jupyter notebooks in each environment.
 
in this work we use the processed data from point85AI git repository that can be found at:  
https://github.com/point85AI/Policy-Iteration-AI-Clinician

the data set we use to construct the dynamic treatment regime can be found at:  
data -> normalized_data.mat  
is should be placed at:  
Dynamic treatment regime -> data -> normalized_data.mat
