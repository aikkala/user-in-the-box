# User-in-the-box

- `conda_env.yml` can be used to create a conda environment
- If you run `pip install -e .` it will hopefully register the mobl_arms environment as a gym environment which you can call from other packages (using `gym.make('UIB:mobl-arms-muscles-v0')`)