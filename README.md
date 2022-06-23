# User-in-the-box

This repository contains source code for advanced biomechanical user simulation. The source code provides a flexible (modular) approach to define and model/simulate biomechanical models, perception models, and interaction tasks in MuJoCo. The created models/simulations are designed to run as standalone packages that can be easily shared with others


## Implementation

The **[uitb.simulator.Simulator](https://github.com/aikkala/user-in-the-box/blob/main/uitb/simulator.py)** class defines an OpenAI Gym environment. The **Simulator** consists of three main components: 1) an interaction task (defined as a MuJoCo xml file), 2) a biomechanical model (defined as a MuJoCo xml file), and 3) a perception model (defined as a set of perception modules). All three are wrapped as Python classes in respective modules. In order to add new tasks, biomechanical models, or perception modules, one needs to create new Python modules/classes that inherit from their respective base classes (see below for further instructions).

[TODO maybe a figure here to illustrate the architecture]


### Interaction tasks

Interaction tasks are defined in [uitb/tasks](https://github.com/aikkala/user-in-the-box/tree/main/uitb/tasks). When creating new tasks, one must inherit from the base class **[uitb.tasks.base.BaseTask](https://github.com/aikkala/user-in-the-box/blob/main/uitb/tasks/base.py)**.


### Biomechanical models

The biomechanical models are defined in [uitb/bm_models](https://github.com/aikkala/user-in-the-box/tree/main/uitb/bm_models). When creating new biomechanical models, one must inherit from the base class **[uitb.bm_models.base.BaseBMModel](https://github.com/aikkala/user-in-the-box/tree/main/uitb/bm_models/base.py)**.


### Perception models

The perception modules are defined in [uitb/perception](https://github.com/aikkala/user-in-the-box/blob/main/uitb/perception)/[modality], where [modality] refers to a specific modality like "vision" or "proprioception". Note that this extra [modality] layer will be probably removed in the future. The base class that must be inherited when creating new perception modules is **[uitb.perception.base.BaseModule](https://github.com/aikkala/user-in-the-box/blob/main/uitb/perception/base.py)**.


### Building a Simulator

A Simulator is built according to a _config file_ (which is in yaml format). For examples of such config files see [uitb/configs](https://github.com/aikkala/user-in-the-box/blob/main/uitb/configs). TODO provide a convenience constructor for Simulator that takes the path to a config file as input argument.

The Simulator is built into folder *output/config["run_name"]*, which is a standalone package that contains all the necessary codes to run the Simulator.

```python
from uitb import Simulator
from ruamel.yaml import YAML

# Load a config file
yaml = YAML()
with open(*path/to/config/file.yaml*, 'r') as stream:
  config = yaml.load(stream)

# Build the simulator
run_folder = Simulator.build(config)

```

### Running a Simulator

Once a Simulator has been built as shown above, you can initialise and run the simulator with `simulator = Simulator.get(run_folder)`, where `simulator` can be run in the same way as any OpenAI Gym environment (i.e. by calling methods `simulator.step(action)`, `simulator.reset()`). In addition to using the `Simulator.get(run_folder)` method, one can initialise a Simulator with `gym.make(config["gym_name"])` IF the built simulator folder is in one's Python path and has been imported. config["gym_name"] is automatically added into the config during building, and is simply "uitb:" + config["run_name"] + "-v0" to satisfy OpenAI Gym's naming conventions.   


## Setup

- The conda environment defined in `conda_env.yml` should contain all required packages. Create a new conda env with `conda env create -f conda_env.yml` and activate it with `conda activate uitb`.

- Tested only with EGL for headless rendering: enable with `export MUJOCO_GL=egl`


## Training

The script [uitb/train/trainer.py](https://github.com/aikkala/user-in-the-box/blob/main/uitb/train/trainer.py) takes as a input a config file, then calls `Simulation.build(config)` to build the simulation, and then starts running the training using stable-baselines3. Other RL libraries can be defined in [uitb/rl](https://github.com/aikkala/user-in-the-box/blob/main/uitb/rl), and they must inherit from the base class **[uitb.rl.base.BaseRLModel](https://github.com/aikkala/user-in-the-box/blob/main/uitb/rl/base.py)**. Weights & Biases is used for logging.


## Testing

One can use the script [uitb/test/evaluator.py](https://github.com/aikkala/user-in-the-box/blob/main/uitb/test/evaluator.py) to evaluate the performance of a trained Simulator.


## TODO list
- cameras, lighting
- getting simulation state
- create a setup.py for the simulators so they (and required packages) can be easily installed?


## Troubleshooting
