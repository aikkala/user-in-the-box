## INSTALL UitB-Sim2VR
1. Create environment (venv or virtualenv is possible as well):
	`conda create --name uitb_env python`
	`conda activate uitb_env`
2. Install user-in-the-box and dependencies
	`git clone <repo-url>`
	`cd user-in-the-box/`
	`pip install -e .`
3. (Optional) Install kernel to run Jupyter notebook files for testing and debugging
	`pip install ipykernel`
	`python -m ipykernel install --name uitb_env --user`
4. Prepare environment variables (Linux Bash)
	`echo 'export MUJOCO_GL=egl' >> ~/.bashrc`
	`source ~/.bashrc`

### Run via SSH on remote display
- make sure that $DISPLAY is set to ':0' or ':1' if remote desktop should be used (list of running user sessions: `w -oush)
- TODO: is a non-remote GUI session required to run in background?
- TODO: do we need to run `xhost +local:` on remote workstation?
