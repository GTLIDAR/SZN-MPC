
## Code will be updated
This code is based on the do-mpc library https://www.do-mpc.com/en/latest/
## Structure
The examples folder contains a number of different examples with different models and constraints setup. Each example contains three main codes. main_offline.py; runs the MPC loops, sets initial state, saves the results. template_mpc.py; sets cost function, time varying paramters (pedestrians positions, terrain, obstacles, goals), MPC horizion, and sets the constrains. template_model.py; system model, system states, constraint equations. 

### examples

* decoupled 
    * SZN-MPC decoupled 
    * decoupled_cbb for hardware testing in CCB

* coupled 
    * SZN-MPC coupled

* 2digit_2_quad_LTL
    * Search and rescue with slugs implementation of a team with 2 digits and 2 quadrotors. With sparse GP implemntation for terrain and belief GP


## Running the code
run python main_offline.py, it will save the results in the results folder, after it runs for the specified number of steps. animate.py will vizualize the results.

