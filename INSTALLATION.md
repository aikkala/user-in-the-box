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
- make sure that $DISPLAY is set to ':0' or ':1' if remote desktop should be used (a list of running user sessions can be obtained via `w -oush`)

### Run via SSH on virtual display (using Xvfb)
- create virtual display and let it run in background
-- `export DISPLAY=:1`
-- `xdpyinfo -display $DISPLAY > /dev/null || Xvfb $DISPLAY -screen 0 1920x1090x24 &`
