#Sim2VR Unity Plugin

## Installation 
### Unity
1. Import package "sim2vr.unitypackage"
2. Add prefabs as gameobjects to the scene in which you want to add the simulated user
3. Create a subclass of "RLEnv" in the UserInTheBox namespace which implements (@override) all abstract methods. Those are:
- CalculateReward():
	Defines the reward for the RL agent. You can use for example in-game points, target distance, completion time (usually  as negative reward) or combinations.
	To be compatible with the effort term added by the biomechanical model, the reward should be in range [-x,x] %TODO: find x (mean reward during one task)
- Reset():
	Resets the application, e.g., restarts the level and resets the points, user/target position, etc.
- InitialiseReward():
	Sets the initial reward components, such as initial points, controller position, etc.
- UpdateIsFinished():
	Should set the _isFinished flag to True when the task is finished/aborted. Usually this will then automatically trigger a reset of the agent and environment.
	
You may add optional attributes such as gameobjects that define the application logic to this class. This allows accessing application information such as points easily.
4. Link your subclass script to the "RLEnv" gameobject.	
5. Build your application into the directory "\user-in-the-box\uitb\tasks\unity\application_name" 
6. Create a config.yaml (similar to the sim2vr_example.yaml) %TODO: Build example yaml#
Important components are
- "bm_model/kwargs/skull_rotation":
	Defines the rotation of the biomechanical model %TODO: is it really necessary to set this manually? Set automatically from unity data.
- "unity_executable":
	Should be the path to the unity build executable (OS specific).
- "app_args":
	(Optional) Additional arguments that are passed to the model, such as game settings, e.g., ["-condition", "easy"], ["-level", "3"].
- "override_headset_orientation":
	Necessary if the biomechanical model should look in another direction than straight ahead. 
- "time_scale" (default:5):
	Defines how many times real time Unity tries to run for training.
- "gear" (default: "oculus-quest-1"):
	Defines the head-mounted display that is used.
- "right_controller_body" (default: "hand"):
	%TODO
- "right_controller_relpose": 
	%TODO
- "headset_body" (default: "skull"):
	%TODO
- "headset_relpose":
	%TODO



### Python  




## Troubleshooting
1. Make sure you have installed the XR Interaction Toolkit and activated the "Oculus" Plug-in Provider in the XR Plug-in Management (Project Settings)

