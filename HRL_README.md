## Hierarchical Learning branch
Credits: Rayan Hamid Mohiuddin, Nikita Agrawal

To use our hierarchical approach, you have to mention the following in the config files:

```
simulation:
  run_parameters:
    action_sample_freq: # Has to be set in terms of LLC frequency

rl:
  nsteps: # According to the value to test, by default 4000

# Tells the simulator.py to use HRL
llc:
  llc_ratio: # Number of llc steps per hlc step (Usually 1-20)
  simulator_name: mobl_arms_llc
  checkpoint: "model_270000000_steps.zip" # LLC checkpoint, has to be placed in mobl_arms_llc/checkpoints
  joints: [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion, pro_sup]
```

To train with our evaluations of mean distance to target:

```python
 python uitb/train/trainer.py uitb/configs/mobl_arms_index_task.yaml --eval --eval_info_keywords dist_from_target
```

To train the different models we implemented, do the following:

1. 1:1 Pointing 20/40 Hz
- Change action_sample_freq: 20 or 40
- Set llc_ratio: 1
- run:
```
python uitb/train/trainer.py uitb/configs/mobl_arms_index_pointing.yaml --eval --eval_info_keywords dist_from_target
```
2. 1:20 Pointing 400 Hz
- Change action_sample_freq: 400
- Set llc_ratio: 20
- Optional: save_freq: 1_000_000 # So you can receive eval checkpoints faster
- Optional: nsteps: 200, 1000, 2000, 4000
- run:
```
python uitb/train/trainer.py uitb/configs/mobl_arms_index_pointing.yaml --eval --eval_info_keywords dist_from_target
```
3.  1:1 Choice Reaction 20/40 Hz
- Change action_sample_freq: 20 or 40
- Set llc_ratio: 1
- run:
```
python uitb/train/trainer.py uitb/configs/mobl_arms_index_choice_reaction.yaml --eval --eval_info_keywords dist_from_target
```
4.  1:20 Choice Reaction 20, 40, 100, 400 Hz
- Change action_sample_freq: 0 or 40 or 100 or 400 Hz
- Set llc_ratio: 20
- Optional: save_freq: 1_000_000 # So you can receive eval checkpoints faster
- Optional: nsteps: 200 or 1000 or 2000 or 4000
- run:
```
python uitb/train/trainer.py uitb/configs/mobl_arms_index_choice_reaction.yaml --eval --eval_info_keywords dist_from_target
```

To use accumulated llc step rewards, uncomment lines 430 and 458 in simulator.py and replace line 486 with following:
```python
        return obs, acc_reward, terminated, truncated, info
```

To evaluate the different models we implemented:
1. 1:1 Pointing 20/40 Hz
- Set llc_ratio: 1
- run:
```
python uitb/test/evaluator.py simulators/mobl_arms_index_pointing --checkpoint pointing1_1_20hz/model_65000000_steps --num_episodes 10 --record --logging --action_sample_freq 100
```

or

```
python uitb/test/evaluator.py simulators/mobl_arms_index_pointing --checkpoint pointing1_1_40hz/model_55000000_steps --num_episodes 10 --record --logging --action_sample_freq 100
```


2. 1:20 Pointing 400 Hz
- Set llc_ratio: 20
- run:
```
python uitb/test/evaluator.py simulators/mobl_arms_index_pointing --checkpoint pointing1_20_400hz/model_20000000_steps --num_episodes 10 --record --logging --action_sample_freq 400
```

3.  1:1 Choice Reaction 20/40 Hz
- Set llc_ratio: 1
- run:
```
python uitb/test/evaluator.py simulators/mobl_arms_index_choice_reaction --checkpoint choice_reaction1_1_20hz/model_20000000_steps --num_episodes 10 --record --logging --action_sample_freq 100
```

or

```
python uitb/test/evaluator.py simulators/mobl_arms_index_choice_reaction --checkpoint choice_reaction1_1_40hz/model_14000000_steps --num_episodes 10 --record --logging --action_sample_freq 100

```
4.  1:20 Choice Reaction 400 Hz
- Set llc_ratio: 20
- run:
```
python uitb/test/evaluator.py simulators/mobl_arms_index_choice_reaction --checkpoint choice_reaction1_20_400hz/model_6000000_steps --num_episodes 10 --record --logging --action_sample_freq 400
```

The above runs the pre-trained simulator for 10 episodes, records videos and saves log files of the evaluted episodes, and samples actions with a frequency of 100 Hz from the policy. The videos and log files will  be saved inside `simulators/mobl_arms_task/evaluate` folder. Run `python uitb/test/evaluator.py --help` for more information.
