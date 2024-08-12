from .simulator import Simulator

from gymnasium.envs.registration import register
import pathlib

module_folder = pathlib.Path(__file__).parent
simulator_folder = module_folder.parent
kwargs = {'simulator_folder': simulator_folder}
register(id=f'{module_folder.stem}-v0', entry_point=f'{module_folder.stem}.simulator:Simulator', kwargs=kwargs)
