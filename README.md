# Breathing Life Into Biomechanical Models UIST2022

This branch contains the trained models, generated data, and analysis scripts required to replicate the figures presented in the UIST2022 paper titled "Breathing Life Into Biomechanical Models"

Note that the code in this branch should not be used for further research and development of the User-in-the-Box software, as it is not up-to-date. Instead, go to the [main branch](https://github.com/aikkala/user-in-the-box/tree/main) of this repository.

## Models

Below you can find the trained models which were analysed in the paper.

| **Task**                           | **Folder**                                           | **Config name**                  |
|------------------------------------|----------------------------------------------------------|-----------------------------|
| Pointing & ISO Pointing            | [UIST/iso-pointing-U1-patch-v1-dwell-random](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIST/iso-pointing-U1-patch-v1-dwell-random/)                    | mobl_arms_iso_pointing_uist |
| Tracking                           | [UIST/tracking-v1-patch-v1](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIST/tracking-v1-patch-v1/)                                     | mobl_arms_tracking_uist     |
| Button Press                       | [UIST/button-press-v1-patch-v1-smaller-buttons](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIST/button-press-v1-patch-v1-smaller-buttons/)                 | mobl_arms_button_press_uist |
| Controlling an RC Car Via Joystick | [UIST/driving-model-v2-patch-v1-newest-location-no-termination](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIST/driving-model-v2-patch-v1-newest-location-no-termination) | mobl_arms_remote_driving_uist |

The second column indicates the folder name for each task. The trained model files are located in subfolders called **checkpoints**.

You can train the models from scratch by using the script [trainer.py](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIB/train/trainer.py). Inside the script you must define a config (line 14), and the third column in above table indicates the variable names of the configs that were used to train the models. The configs are defined in [UIB/train/configs.py](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIB/train/configs.py). 


## Data 

The data files that were used in the paper are found in subfolders called **evaluate**.

You can also collect the data again with the trained models by using the script [evaluator.py](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIB/test/evaluator.py) and [repeated-movements.py](https://github.com/aikkala/user-in-the-box/blob/uist-submission-aleksi/UIB/test/mobl_arms/pointing/repeated-movements.py). The latter script is needed to produce the data shown in the bottom half of Figure 5, the data produced by the former script is sufficient to produce other Figures.


## Analysis

TBD
