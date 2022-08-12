# User-in-the-box

- This repository contains source code for modeling and simulating HCI interaction tasks in [MuJoCo](https://mujoco.org/). The user is modeled with a muscle-actuated biomechanical model with perception capabilities (e.g. egocentric vision), and is trained with reinforcement learning to solve the interaction task.
- The source code provides a flexible (modular) approach to implementing new biomechanical models, perception models, and interaction tasks.
- The produced models/simulations (which will be henceforth referred to as *simulators*) are designed to run as standalone packages that can be easily shared with others. These simulators inherit and implement the OpenAI Gym interface, and hence they are easy to use and e.g. can be easily plugged into existing RL training libraries.


## Video

https://user-images.githubusercontent.com/7627254/184347198-2d7f8852-d50b-457f-8eaa-07720b9522eb.mp4
[See here for a Youtube version (with subtitles)](https://youtu.be/-L2hls8Blyc)


## Implementation

The main entry point is **[uitb.simulator.Simulator](https://github.com/aikkala/user-in-the-box/blob/main/uitb/simulator.py)** class, which implements an OpenAI Gym interface. A *simulator* (instance of the Simulator class) consists of three main components: 1) an interaction task (defined as a MuJoCo xml file wrapped in a Python class), 2) a biomechanical model (defined as a MuJoCo xml file wrapped in a Python class), and 3) a perception model (defined as a set of *perception modules*, which are Python classes). In order to add new tasks, biomechanical models, or perception modules, one needs to create new Python modules/classes that inherit from their respective base classes (see below for further instructions).

<br/>

| ![Main classes in software architecture](./figs/architecture.svg "Main classes in software architecture") |
|:--:|
| **Figure 1.** This figure shows the three main components and their relations. The white boxes are classes that a user should not need to edit. The yellow boxes are examples of classes that have already been implemented, and the green boxes indicate classes that need to be implemented when adding new models or tasks. The arrows indicate inheritance. The _Perception_ class contains a set of perception modules (e.g. vision, proprioception). Also, the grey boxes under *Perception* indicate a hierarchical folder structure, which are not actual classes themselves. |

<br/>

### Biomechanical models

The biomechanical models are defined in [uitb/bm_models](https://github.com/aikkala/user-in-the-box/tree/main/uitb/bm_models). When creating new biomechanical models, one must inherit from the base class **[uitb.bm_models.base.BaseBMModel](https://github.com/aikkala/user-in-the-box/tree/main/uitb/bm_models/base.py)**. In addition to implementing a Python class, the biomechanical model must be defined as a (standalone) MuJoCo xml file. One option to creating new biomechanical models is to import them from [OpenSim](https://simtk.org/projects/opensim) models with an [OpenSim-to-MuJoCo converter](https://github.com/aikkala/O2MConverter).


### Interaction tasks

Interaction tasks are defined in [uitb/tasks](https://github.com/aikkala/user-in-the-box/tree/main/uitb/tasks). When creating new tasks, one must inherit from the base class **[uitb.tasks.base.BaseTask](https://github.com/aikkala/user-in-the-box/blob/main/uitb/tasks/base.py)**. In addition to implementing a Python class, the interaction task must be defined as a (standalone) MuJoCo xml file.


### Perception models

A perception model is composed of a set of perception modules, where each module is a specific perception capability, such as vision (egocentric camera) or proprioception (positions, speeds, accelerations of the biomechanical model's joints etc.). The perception modules are defined in [uitb/perception](https://github.com/aikkala/user-in-the-box/blob/main/uitb/perception)/[modality], where [modality] refers to a specific modality like "vision" or "proprioception". Note that this extra [modality] layer might be removed in the future. The base class that must be inherited when creating new perception modules is **[uitb.perception.base.BaseModule](https://github.com/aikkala/user-in-the-box/blob/main/uitb/perception/base.py)**.


### Building a simulator

A simulator is built according to a _config file_ (in yaml format), which defines which models are selected and integrated together to create the simulator. For examples of such config files see [uitb/configs](https://github.com/aikkala/user-in-the-box/blob/main/uitb/configs). In a nutshell, the build process contains two phases. Firstly, the MuJoCo xml file that defines the biomechanical model is integrated into the MuJoCo xml file that defines the interaction task environment, and hence a new standalone MuJoCo xml file is created. Secondly, wrapper classes are called to make sure everything are initialised correctly (e.g. a class inheriting from `BaseTask` might need to move an interaction device to a proper position with respect to a biomechanical model). These initialisations must be defined in the constructors of wrapper classes.

TODO provide a convenience method for building a simulator that takes the path to a config file as input argument.

The simulator is built into folder *[project_path/]simulators/config["simulator_name"]*, which is a standalone package that contains all the necessary codes to run the simulator (given that required Python packages are installed), and hence can be easily shared with others. 

**Note that the simulator name (defined in the config file) should be a valid Python package name (e.g. use underscores instead of dashes).**

```python
from uitb import Simulator

# Define the path to a config file
config_file = "/path/to/config_file.yaml"

# Build the simulator
simulator_folder = Simulator.build(config_file)

```

### Running a simulator

Once a simulator has been built as shown above, you can initialise the simulator with 

```python
simulator = Simulator.get(simulator_folder)
```

and `simulator` can be run in the same way as any OpenAI Gym environment (i.e. by calling methods `simulator.step(action)`, `simulator.reset()`). In addition to the above initialisation method, one can initialise a simulator with (e.g. if config["simulator_name"] = "mobl_arms_index_pointing") 

```python
# Import the simulator and OpenAI Gym
import mobl_arms_index_pointing
import gym

# Initialise a simulator
simulator = gym.make(config["gym_name"])
```

**IF** the simulator folder is in Python path. `config["gym_name"]` is automatically added into the config during building, and is simply `"uitb:" + config["simulator_name"] + "-v0"` to satisfy OpenAI Gym's naming conventions. Alternatively, you can programmatically import a simulator with 

```python
# Add simulator_folder to Python path
import sys
sys.path.insert(0, simulator_folder)
```


## Setup

- The conda environment defined in `conda_env.yml` should contain all required packages. Create a new conda env with `conda env create -f conda_env.yml` and activate it with `conda activate uitb`.

- Tested only with EGL for headless rendering: enable with `export MUJOCO_GL=egl`


## Training

The script [uitb/train/trainer.py](https://github.com/aikkala/user-in-the-box/blob/main/uitb/train/trainer.py) takes as a input a config file, then calls `Simulation.build(config)` to build the simulator, and then starts running the RL training using stable-baselines3. Other RL libraries can be defined in [uitb/rl](https://github.com/aikkala/user-in-the-box/blob/main/uitb/rl), and they must inherit from the base class **[uitb.rl.base.BaseRLModel](https://github.com/aikkala/user-in-the-box/blob/main/uitb/rl/base.py)**. Weights & Biases is used for logging.

Note that you need to define a reward function when creating new interaction tasks. The implementation details of the reward function are (at least for now) left for users to decide. 


## Testing

One can use the script [uitb/test/evaluator.py](https://github.com/aikkala/user-in-the-box/blob/main/uitb/test/evaluator.py) to evaluate the performance of a trained simulator. (WIP)


## Pre-trained simulators
TODO. These simulators correspond to the ones used in the publications. Note: the software architecture has changed since submission, but these are more or less equivalent. Links to original models used in the publication. Links to new models (MoBL ARMS model used in pointing, tracking, choice reaction, remote driving). These simulators can be created and trained again using the existing config files.


## TODO list
- cameras, lighting
- separate Task class into World and Task classes, where the former defines the world and the latter defines only the interactive task?
- create a setup.py for the simulators so they (and required packages) can be easily installed?


## Troubleshooting
No currently known issues.

## Cite
TBD

## Contributors
Aleksi Ikkala  
Florian Fischer  
Markus Klar  
Arthur Fleig  
Miroslav Bachinski  
Andrew Howes  
Perttu Hämäläinen  
Jörg Müller  
Roderick Murray-Smith  
Antti Oulasvirta  
