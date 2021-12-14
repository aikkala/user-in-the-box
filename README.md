# User-in-the-box

- You'll need to install at least mujoco_py, stable_baselines3, openai gym, torch (I'll try to create a conda env for this in the future, my current env has way too much unnecessary packages)
- If you run `pip install -e .` it will hopefully register the mobl_arms environment as a gym environment which you can call from other packages (using `gym.make('UIB:mobl-arms-muscles-v0')`)