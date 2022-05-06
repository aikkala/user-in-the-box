# User-in-the-box

- `conda_env.yml` can be used to create a conda environment


## The Idea

This toolbox aims to create a flexible modelling approach for various different interaction tasks with different user models. 

## The Simulator class

This class defines an OpenAI gym environment "UIB:simulator-v0", which takes as an input a path to a simulation folder. The simulation folder contains all necessary parts for the simulation: a biomechanical model, a perception model, and the task model. The simulation folder is created with class method Simulation.build(config), where the config (defined in UIB/train/configs.py) defines the models.

## Biomechanical models

Defined in UIB/bm_models

## Perception models

Defined in UIB/perception

## Task models

Defined in UIB/tasks

The created simulation folder is supposed to be a standalone package that can be easily shared with others. I'm not sure if we can make it completely standalone, and if the approach I'm using makes sense (essentially clone the original classes). Might make more sense to just use docker or singularity.

## Training

The script UIB/train/trainer.py takes as a input a config file, then calls Simulation.build(config) to build the simulation, and then starts running the training. 

## TODO list
- reward functions
- effort functions
- cameras, lighting
- getting simulation state
- evalutions of trained policies


## Troubleshooting

### GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'

Follow the reply [given here](https://github.com/openai/mujoco-py/issues/172#issuecomment-680701806): set the environment variable LD_PRELOAD as `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` 