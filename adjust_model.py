#import xmltodict
import xmltodict_keeporder as xmltodict  #fork that keeps order of child elements (required for parsing MJCF files)

import gym
import mujoco_py

from utils import opensim_file, adjust_mujoco_model_pt1, adjust_mujoco_model_pt2, print_musculotendon_properties
from utils import adjust_mujoco_model_pt0
from utils import compare_MuJoCo_OpenSim_models

if __name__=="__main__":
    env_name = 'UIB:mobl-arms-muscles-v0'

    opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41.osim"
    #opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_module6_7_CMC.osim"
    mujoco_input = "UIB/envs/mobl_arms/models/mobl_arms_muscles.xml"
    mujoco_intermediate = "UIB/envs/mobl_arms/models/mobl_arms_muscles_tendonwrapping.xml"
    mujoco_output = "UIB/envs/mobl_arms/models/mobl_arms_muscles_modified.xml"
    model_properties_output = "UIB/envs/mobl_arms/models/MoBL_ARMS_analysis.xml"

    # mujoco_input = "UIB/envs/mobl_arms/models/mobl_arms_muscles_original.xml"
    # opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41_fixed_hand.osim"

    # mujoco_input = "UIB/envs/mobl_arms/models/mobl_arms_muscles_fixedjoints.xml"
    # opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41_fixedjoints.osim"

    # ## TEST WITH MIROSLAV'S NEW HAND MODEL:
    # mujoco_input = "/home/florian/reacher_sg/HandModel/HandModelTSLAdjusted_modified4conversion_converted/HandModelTSLAdjusted_modified4conversion_converted.xml"
    # opensim_input = "/home/florian/reacher_sg/HandModel/HandModelTSLAdjusted_modified4conversion.osim"
    # mujoco_intermediate = "/home/florian/reacher_sg/HandModel/HandModelTSLAdjusted_modified4conversion_converted/HandModelTSLAdjusted_modified4conversion_converted_tendonwrapping.xml"
    # mujoco_output = "/home/florian/reacher_sg/HandModel/HandModelTSLAdjusted_modified4conversion_converted/HandModelTSLAdjusted_modified4conversion_converted_modified.xml"

    # Read and parse OpenSim model
    osim_file = opensim_file(opensim_input)

    # Read and parse MuJoCo model
    with open(mujoco_input, "r") as f:
        text = f.read()
    mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    # Adjust musculotendon properties of MuJoCo model (pt0; via xml file)
    adjust_mujoco_model_pt0(mujoco_xml, osim_file)
    # # Store new (intermediate) MuJoCo model
    # mujoco_xml['mujoco']['compiler']['lengthrange']['@mode'] = "none"  # disable automatic length range computation at compile time, since we compute them manually below
    # out = xmltodict.unparse(mujoco_xml, pretty=True, ordered_mixed_children=True)
    # with open(mujoco_intermediate, 'w') as f:
    #     f.write(out)
    # # Read and parse MuJoCo model
    # with open(mujoco_intermediate, "r") as f:
    #     text = f.read()
    # mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    # Adjust musculotendon properties of MuJoCo model (pt1; via xml file)
    #adjust_mujoco_model_pt1(mujoco_xml, osim_file) #TODO: fix this!

    # Store new (intermediate) MuJoCo model
    mujoco_xml['mujoco']['compiler']['lengthrange']['@mode'] = "none"  # disable automatic length range computation at compile time, since we compute them manually below
    out = xmltodict.unparse(mujoco_xml, pretty=True, ordered_mixed_children=True)
    with open(mujoco_intermediate, 'w') as f:
        f.write(out)

    # Adjust musculotendon properties of MuJoCo model (pt2; via gym env)
    env = gym.make(env_name, xml_file=mujoco_intermediate.split("envs/mobl_arms/")[-1], sample_target=False)
    adjust_mujoco_model_pt2(env, osim_file)

    # Store new MuJoCo model
    mujoco_py.cymj._mj_saveLastXML(mujoco_output, env.sim.model, "NULL", 0)
    print(f"MuJoCo model successfully adjusted and stored at\n{mujoco_output}")

    # Store musculotendon properties of both adjusted MuJoCo model and reference OpenSim model in text file
    print_musculotendon_properties(env, osim_file, stdout_file=model_properties_output)

    #env = gym.make(env_name, xml_file="models/mobl_arms_muscles_original.xml", sample_target=False)
    #opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41_fixed_hand.osim"
    #env = gym.make(env_name, xml_file="models/mobl_arms_muscles_fixedjoints.xml", sample_target=False)
    #opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41_fixedjoints.osim"
    #env = gym.make(env_name, xml_file="models/mobl_arms_muscles_fixedjoints2.xml", sample_target=False)
    #opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41_fixedjoints2.osim"
    ##env = gym.make(env_name, xml_file="models/SimpleArm.xml", sample_target=False)
    ##opensim_input = "UIB/envs/mobl_arms/models/SimpleArm.osim"

    #env = gym.make("gym_SG:ReacherSGbasic-v0")  #for testing "arm_hand_SG_osim_coordinates_complete_v5.3ao.xml"

    compare_MuJoCo_OpenSim_models(env, opensim_input)
