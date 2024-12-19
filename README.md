
The source code on this repository is dependent on the do-mpc library https://www.do-mpc.com/en/latest/


BibTex
```
@article{shamsah2024socially,
  title={Socially acceptable bipedal robot navigation via social zonotope network model predictive control},
  author={Shamsah, Abdulaziz and Agarwal, Krishanu and Katta, Nigam and Raju, Abirath and Kousik, Shreyas and Zhao, Ye},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2024}
}

@article{shamsah2024real,
  title={Real-time Model Predictive Control with Zonotope-Based Neural Networks for Bipedal Social Navigation},
  author={Shamsah, Abdulaziz and Agarwal, Krishanu and Kousik, Shreyas and Zhao, Ye},
  journal={IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2024}
}
```

## Structure
The examples folder contains a number of different examples with different models and constraints setup. Each example contains three main codes. 
* main_offline.py 
    * runs the MPC loop
    * sets initial conditions
    * saves the results

* template_mpc.py 
    * sets cost function
    * time varying paramters (pedestrians positions, terrain, obstacles, goals), MPC horizion, and sets the constrains

* template_model.py
    * sets system model
    * sets system states
    * sets constraints equations

(For more details on the structure of the code visit https://www.do-mpc.com/en/latest/)

### examples

* decoupled 
    * SZN-MPC decoupled 

* coupled 
    * SZN-MPC coupled


## Running the code

run python main_offline.py, it will save the results in the results folder, after it runs for the specified number of steps. 
```bash
cd ~/decoupled
python main_offline.py
```

## Visualization

```bash
cd ~/decoupled
python animate.py
```
