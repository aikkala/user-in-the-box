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
    # opensim_input = "UIB/envs/mobl_arms/models/MOBL_ARMS_module2_4_allmuscles.osim"
    mujoco_input = "UIB/envs/mobl_arms/models/mobl_arms_muscles.xml"
    mujoco_intermediate = "UIB/envs/mobl_arms/models/mobl_arms_muscles_tendonwrapping.xml"
    mujoco_output = "UIB/envs/mobl_arms/models/mobl_arms_muscles_modified.xml"
    model_properties_output = "UIB/envs/mobl_arms/models/MoBL_ARMS_analysis.xml"

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
    adjust_mujoco_model_pt1(mujoco_xml, osim_file)

    # Store new (intermediate) MuJoCo model
    mujoco_xml['mujoco']['compiler']['lengthrange']['@mode'] = "none"  # disable automatic length range computation at compile time, since we compute them manually below
    out = xmltodict.unparse(mujoco_xml, pretty=True, ordered_mixed_children=True)
    with open(mujoco_intermediate, 'w') as f:
        f.write(out)

    # Adjust musculotendon properties of MuJoCo model (pt2; via gym env)
    env = gym.make(env_name, xml_file=mujoco_intermediate.split("envs/mobl_arms/")[-1], sample_target=False)
    adjust_mujoco_model_pt2(env, osim_file)


    ##########


    import time
    start_time = time.time()

    model = mujoco_py.load_model_from_path("/home/florian/user-in-the-box/UIB/envs/mobl_arms/models/test.xml")
    print(env.sim.model.tex_rgb, model.tex_rgb, sum(model.tex_rgb == 255), len(model.tex_rgb))
    #from mujoco_py.utils import rec_assign, rec_copy
    #rec_assign(env.sim.model.tex_rgb, rec_copy(model.tex_rgb))
    import sys, ctypes
    def mutate(obj,
               new_obj):  # from https://stackoverflow.com/questions/49841577/updating-a-variable-by-its-id-within-python
        if sys.getsizeof(obj) != sys.getsizeof(new_obj):
            raise ValueError('objects must have same size')
        mem = (ctypes.c_byte * sys.getsizeof(obj)).from_address(id(obj))
        new_mem = (ctypes.c_byte * sys.getsizeof(new_obj)).from_address(id(new_obj))
        for i in range(len(mem)):
            mem[i] = new_mem[i]
    from mujoco_py.modder import TextureModder
    modder = TextureModder(env.sim)
    print([i.tex_rgb for i in modder.textures])
    for render_context in env.sim.render_contexts:
        render_context.upload_texture(0)

    mutate(env.sim.model.tex_rgb, model.tex_rgb)
    # mujoco_py.cymj._mjr_uploadTexture(env.sim.model, mujoco_py.cymj.MjRenderContextWindow, 0)
    for render_context in env.sim.render_contexts:
        render_context.upload_texture(0)
    print(env.sim.model.tex_rgb, model.tex_rgb, sum(model.tex_rgb == 255), len(model.tex_rgb))
    input([i.tex_rgb for i in modder.textures])

    import _ctypes
    input((model.uintptr))
    import pyximport

    pyximport.install()
    import buf_test

    data = _ctypes.PyObj_FromPtr(id(model.uintptr))
    data = model.qpos_spring
    buf_test.test_double(data)  # works fine - but interprets as unsigned char
    # we can also use casts and the Python
    # standard memoryview object to get it as a double array
    buf_test.test_double(memoryview(data).cast('d'))

    mutate(model.qpos0, model.qpos_spring)
    # from mujoco_py.utils import rec_assign, rec_copy
    # rec_assign(model.qpos_spring, rec_copy(model.qpos0))
    model.qpos_spring[:] = [7] * model.nq
    print(model.qpos0, model.qpos_spring)
    input(_ctypes.PyObj_FromPtr(model.uintptr))
    print(id(model.qpos0), id(model.qpos_spring))
    input(time.time() - start_time)
    ##########

    # Store new MuJoCo model
    mujoco_py.cymj._mj_saveLastXML(mujoco_output, env.sim.model, "NULL", 0)
    print(f"MuJoCo model successfully adjusted and stored at\n{mujoco_output}")

    # Store musculotendon properties of both adjusted MuJoCo model and reference OpenSim model in text file
    print_musculotendon_properties(env, osim_file, stdout_file=model_properties_output)

    compare_MuJoCo_OpenSim_models(env, opensim_input)
