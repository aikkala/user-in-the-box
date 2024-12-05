## INSTALL UitB-Sim2VR
1. Create conda environment (venv or virtualenv is possible as well)
- `conda create --name uitb_env python`
- `conda activate uitb_env`
2. Install user-in-the-box and dependencies
- `git clone <repo-url>`
- `cd user-in-the-box/`
- `pip install -e .`
3. (Optional) Install kernel to run Jupyter notebook files for testing and debugging
- `pip install ipykernel`
- `python -m ipykernel install --name uitb_env --user`
4. Prepare environment variables (Linux Bash)
- `echo 'export MUJOCO_GL=egl' >> ~/.bashrc`
- `source ~/.bashrc`
- `conda activate uitb_env`

### Run via SSH on remote display
- make sure that `$DISPLAY` is set to ':0' or ':1' if remote desktop should be used (a list of running user sessions can be obtained via `w -oush`)
- TODO: check if `xhost +local:` is required

### Run via SSH on virtual display (using Xvfb; only works on Linux!)
- create virtual display (the 1 below is just a placeholder for any previously unused display ID) and let it run in background
  - `export DISPLAY=:1`
  - `xdpyinfo -display $DISPLAY > /dev/null || Xvfb $DISPLAY -screen 0 1920x1090x24 &`
- to check which display IDs are already in use (e.g., on an HPC):
  - `ls -l /tmp/.X11-unix/`

### Troubleshooting
- Error: XDG_RUNTIME_DIR is invalid or not set in the environment
  - set this env variable permanently (TODO: verify that this fixes the issue)
    - `echo "export XDG_RUNTIME_DIR=/run/user/$(id -u)" >> ~/.bashrc`
    - `source ~/.bashrc`
