#import xmltodict
import os.path

import xmltodict_keeporder as xmltodict  #fork that keeps order of child elements (required for parsing MJCF files)
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import sys
import opensim as osim  #required to compute fpmax, and to run OpenSim forward simulations
from gym import spaces  #required to run OpenSim forward simulations
import yaml  #for pretty print
import pandas as pd
from opensim_static_optimization import static_optimization
import mujoco_py

class opensim_file(object):
  """
  Object class that parses .osim-file and stores all occurrences of a given xml-tag as Python dict
  in the corresponding class attribute (at runtime, if not called before).
  """
  def __init__(self, filepath):
    # Read filepath and parse it
    with open(filepath) as f:
      text = f.read()
    self.opensim_xml = xmltodict.parse(text)
    self.osim_version = self.opensim_xml["OpenSimDocument"]["@Version"]
    for setname in ["ForceSet", "BodySet", "ConstraintSet", "MarkerSet", "ContactGeometrySet"]:
      setattr(self, setname, self.opensim_xml["OpenSimDocument"]["Model"][setname])
    self.BodySet["objects"]["Body"].append(self.opensim_xml["OpenSimDocument"]["Model"]["Ground"])

  def __getattr__(self, name):
    for typeset in ["ForceSet", "BodySet", "ConstraintSet", "MarkerSet", "ContactGeometrySet"]:
        keyqueue_list = self._itemexists(getattr(self, typeset), name)
        if keyqueue_list:
            if isinstance(keyqueue_list,str):
                keyqueue_list = [keyqueue_list]
            newattr = self._valuesdict(name, typeset, keyqueue_list)
            setattr(self, name, newattr)  #avoid re-computation of this attribute
            return newattr

  def _itemexists(self, obj, key, keyqueue="", ignorekeys=["Ligament"]):
    if key in obj:
        return key
    for k, v in obj.items():
        if k in ignorekeys:
            continue
        if isinstance(v,dict):
            returnvalue = self._itemexists(v, key, keyqueue)
            if isinstance(returnvalue,str):
                return k+"@"+v["@name"]+"/"+returnvalue if ("@name" in v) else k+"/"+returnvalue
            elif isinstance(returnvalue,list):
                return [k+"@"+v["@name"]+"/"+returnvalue_instance if ("@name" in v) else k+"/"+returnvalue_instance for returnvalue_instance in returnvalue]
        elif isinstance(v,list):
            keyqueue_list = []
            for idx, w in enumerate(v):
                if isinstance(w,dict):
                    returnvalue = self._itemexists(w, key, keyqueue)
                    if isinstance(returnvalue,str):
                        keyqueue_list.append(k+"@"+w["@name"]+"/"+returnvalue)
                    elif isinstance(returnvalue,list):
                        for returnvalue_instance in returnvalue:
                            keyqueue_list.append(k+"@"+w["@name"]+"/"+returnvalue_instance)
            if len(keyqueue_list) > 0:
                return keyqueue_list

  def _valuesdict(self, name, typeset, keyqueue_list, readable_keys=True):
    """Return all values of name in some set."""
    #input(keyqueue_list)
    values_dict = {}
    for keyqueue in keyqueue_list:
        typeset_dict = getattr(self, typeset)
        name_ids = []
        for key in keyqueue.split("/"):
            if "@" in key:
                key0, name_id = key.split("@")
                name_ids.append(name_id)
                if isinstance(typeset_dict[key0],list):
                    typeset_dict = [val for val in typeset_dict[key0] if val["@name"] == name_id]
                    assert len(typeset_dict) == 1
                    typeset_dict = typeset_dict[0]
                elif isinstance(typeset_dict[key0],dict):
                    typeset_dict = typeset_dict[key0]
                else:
                    raise TypeError
            else:
                typeset_dict = typeset_dict[key]
        if readable_keys:
            values_dict["/".join(name_ids)] = typeset_dict
        else:
            values_dict[keyqueue] = typeset_dict
    return values_dict


def adjust_mujoco_model_pt0(mujoco_xml, osim_file):
  """
  Adjusts the musculotendon properties of a MuJoCo model, using the corresponding OpenSim model as reference.
  Pt. 0: Sets tendon spatial properties (damping, stiffness, and springlength),
  and (re-)adds all tendon path points from the OpenSim model as sites to the MuJoCo model.
  :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
  :param osim_file: OpenSim model (filepath or parsed opensim_file class)
  :return: None [if mujoco_xml is of type dict, it is updated inplace; the model file is not overwritten here!]
  """

  if type(osim_file) == str:
    osim_file = opensim_file(osim_file)

  # MUJOCO MODEL ADJUSTMENT: Align thorax center with origin of global system:
  try:
    get_mujoco_body(mujoco_xml, "thorax")["@pos"] = array_to_strinterval(np.zeros((3,)))
    get_mujoco_body(mujoco_xml, "thorax")["@quat"] = array_to_strinterval([1, 0, 0, 0])
  except AssertionError:
    get_mujoco_body(mujoco_xml, "ground")["@pos"] = array_to_strinterval(np.zeros((3,)))
    get_mujoco_body(mujoco_xml, "ground")["@quat"] = array_to_strinterval([1, 0, 0, 0])
  ##-> adjust direction of gravity
  if 'option' not in mujoco_xml['mujoco']:
    mujoco_xml['mujoco']['option'] = []
  mujoco_xml['mujoco']['option']["@gravity"] = array_to_strinterval([0, -9.80665, 0])

  # MUJOCO MODEL ADJUSTMENT: Clear tendon spring-damper properties:
  spatial_tendons = mujoco_xml['mujoco']['tendon']['spatial']
  for spatial_tendon in spatial_tendons:
    if '@springlength' in spatial_tendon:
        spatial_tendon.pop('@springlength')
    if '@damping' in spatial_tendon:
        #spatial_tendon['@damping'] = array_to_strinterval([0.1])
        spatial_tendon.pop('@damping')
  # TODO: set "tendon_stiffness" to "stiffness_at_one_norm_force" from OpenSim model?

  # # Append OpenSim path points as sites to MuJoCo model:
  # #TODO
  #
  # # Reference sites in tendon paths:
  # ## To keep order of sites, we first need to remove all existing sites
  # spatial_tendons = mujoco_xml['mujoco']['tendon']['spatial']
  # for spatial_tendon in spatial_tendons:
  #   if 'site' in spatial_tendon:
  #       spatial_tendon.pop('site')
  #   if 'geom' in spatial_tendon:
  #       spatial_tendon.pop('geom')
  #
  # for key, osim_path_point in {**osim_file.PathPoint, **osim_file.MovingPathPoint, **osim_file.MovingPathPoint}.items():
  #   tendon_name = key.split("/")[0] + '_tendon'
  #   if type(osim_path_point) == list:
  #     for path_point in osim_path_point:
  #       if True: #TODO: #wrap_path['wrap_object'] not in failed_objects:
  #           _append_OpenSim_path_point_to_MuJoCo_model(mujoco_xml, tendon_name, path_point)
  #   else:
  #     if True: #TODO: #osim_wrap_path['wrap_object'] not in failed_objects:
  #         _append_OpenSim_path_point_to_MuJoCo_model(mujoco_xml, tendon_name, osim_path_point)


def adjust_mujoco_model_pt1(mujoco_xml, osim_file):
  """
  Adjusts the musculotendon properties of a MuJoCo model, using the corresponding OpenSim model as reference.
  Pt. 1: Adds tendon wrapping objects and adjusts tendon paths.
  :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
  :param osim_file: OpenSim model (filepath or parsed opensim_file class)
  :return: None [if mujoco_xml is of type dict, it is updated inplace; the model file is not overwritten here!]
  """

  if type(osim_file) == str:
    osim_file = opensim_file(osim_file)

  # Append OpenSim wrap objects to MuJoCo model:
  failed_objects = []
  wrap_object_types = ["WrapCylinder", "WrapSphere", "WrapEllipsoid", "WrapTorus"]
  for object_type in wrap_object_types:
    for key, osim_wrap_object in getattr(osim_file, object_type).items():
      key = key.split("/")[0]
      if key.lower() == "ground":
          key = "thorax"
      current_body = get_mujoco_body(mujoco_xml, key)
      if type(osim_wrap_object) == list:
        for wrap_object in osim_wrap_object:
          if obj_name := append_OpenSim_wrap_object_to_MuJoCo_body(current_body, wrap_object,
                                                                   object_type=object_type): failed_objects.append(
            obj_name)
          # object names are only returned in case of failure to append them to the model
      else:
        if obj_name := append_OpenSim_wrap_object_to_MuJoCo_body(current_body, osim_wrap_object,
                                                                 object_type=object_type): failed_objects.append(
          obj_name)
        # object names are only returned in case of failure to append them to the model

  # Use object geometries as obstacles for tendon paths:
  ## First add geometries only located between some specific sites:
  for key, osim_wrap_path in osim_file.PathWrap.items():
    tendon_name = key.split("/")[0] + '_tendon'
    if type(osim_wrap_path) == list:
      for wrap_path in osim_wrap_path:
        if wrap_path['wrap_object'] not in failed_objects:
          wrap_path_range = wrap_path['range'] if 'range' in wrap_path else "-1 -1"
          if any(strinterval_to_nparray(wrap_path_range) != np.array([-1, -1])):
            append_OpenSim_wrap_path_to_MuJoCo_model(mujoco_xml, tendon_name, wrap_path)
    else:
      if osim_wrap_path['wrap_object'] not in failed_objects:
        osim_wrap_path_range = osim_wrap_path['range'] if 'range' in osim_wrap_path else "-1 -1"
        if any(strinterval_to_nparray(osim_wrap_path_range) != np.array([-1, -1])):
          append_OpenSim_wrap_path_to_MuJoCo_model(mujoco_xml, tendon_name, osim_wrap_path)

  ## Then add remaining geometries wherever possible, and replace above geometries if their size is much smaller:
  for key, osim_wrap_path in osim_file.PathWrap.items():
    tendon_name = key.split("/")[0] + '_tendon'
    if type(osim_wrap_path) == list:
      for wrap_path in osim_wrap_path:
        if wrap_path['wrap_object'] not in failed_objects:
          wrap_path_range = wrap_path['range'] if 'range' in wrap_path else "-1 -1"
          if all(strinterval_to_nparray(wrap_path_range) == np.array([-1, -1])):
            append_OpenSim_wrap_path_to_MuJoCo_model(mujoco_xml, tendon_name, wrap_path)
    else:
      if osim_wrap_path['wrap_object'] not in failed_objects:
        osim_wrap_path_range = osim_wrap_path['range'] if 'range' in osim_wrap_path else "-1 -1"
        if all(strinterval_to_nparray(osim_wrap_path_range) == np.array([-1, -1])):
          append_OpenSim_wrap_path_to_MuJoCo_model(mujoco_xml, tendon_name, osim_wrap_path)

  # Create default classes for each OpenSim wrapping object type:
  wrap_object_rgba = "0.8 0.5 0.4 0"  #"0.5 0 0.5 0"
  default_body = mujoco_xml['mujoco']['default']
  if 'default' not in default_body:
      default_body['default'] = []
  elif type(default_body['default']) == dict:
      default_body['default'] = [default_body['default']]
  for object_type in wrap_object_types:
      default_body['default'].append({'@class': object_type,
                                  'geom': {'@rgba': wrap_object_rgba}})


def adjust_mujoco_model_pt2(env, osim_file, scale_ratio=None):
    """
     Adjusts the musculotendon properties of a MuJoCo model, using the corresponding OpenSim model as reference.
     Pt. 2: Adjusts scale ratios of optimal fiber length, max. isometric force, activation time constants, and actuator length ranges.
     :param env: Gym environment of MuJoCo model
     :param osim_file: OpenSim model (filepath or parsed opensim_file class)
     :param scale_ratio: scale ratios of optimal fiber length (array_like object; default is set below)
     :return: None [env is updated inplace; the model file is not overwritten here!]
     """

    if type(osim_file) == str:
        osim_file = opensim_file(osim_file)

    actuator_indexlist = [i for i in range(env.sim.model.nu) if env.sim.model.actuator_trntype[i] == 3]

    # MUJOCO MODEL ADJUSTMENT: adjust scale ratios of optimal fiber length
    if not scale_ratio:
        # [values taken from Garner and Pandy (2002), https://web.ecs.baylor.edu/faculty/garner/Research/GarnerPandy2003ParamEst.pdf]
        scale_ratio = [0.5, 1.5]
    env.sim.model.actuator_gainprm[:, :2] = [scale_ratio] * env.sim.model.nu

    # MUJOCO MODEL ADJUSTMENT: adjust maximum isometric force
    for actuator_id in actuator_indexlist:  # only consider tendon actuators
        actuator_name = env.sim.model.actuator_id2name(actuator_id)
        env.sim.model.actuator_gainprm[actuator_id, 2] = float(osim_file.max_isometric_force[actuator_name])

    # MUJOCO MODEL ADJUSTMENT: adjust FLV curve properties (lmin, lmax, vmax, fpmax, fvmax)
    if osim_file.osim_version.startswith("4"):
        # for details, see comments in OpenSim file and paper of Millard muscles (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3705831/pdf/bio_135_2_021005.pdf)
        for actuator_id in actuator_indexlist:  # only consider tendon actuators
            actuator_name = env.sim.model.actuator_id2name(actuator_id)
            actuator_aflc_name = f"{actuator_name}/{actuator_name}_ActiveForceLengthCurve"
            actuator_pflc_name = f"{actuator_name}/{actuator_name}_FiberForceLengthCurve"
            actuator_fvc_name = f"{actuator_name}/{actuator_name}_ForceVelocityCurve"
            env.sim.model.actuator_gainprm[actuator_id, 4] = float(osim_file.min_norm_active_fiber_length[actuator_aflc_name])  #lmin
            env.sim.model.actuator_gainprm[actuator_id, 5] = float(osim_file.max_norm_active_fiber_length[actuator_aflc_name])  #lmax
            env.sim.model.actuator_gainprm[actuator_id, 6] = float(osim_file.max_contraction_velocity[actuator_name])  #vmax
            # fpmax is not stored in OpenSim file and needs to be computed as passive-force-length curve value at lmax:
            osim_pflc_args = [float(getattr(osim_file, osim_pflc_property)[actuator_pflc_name]) for osim_pflc_property in [
                "strain_at_zero_force", "strain_at_one_norm_force", "stiffness_at_low_force", "stiffness_at_one_norm_force", "curviness"]]
            osim_pflc = osim.FiberForceLengthCurve(*osim_pflc_args)
            assert abs(osim_pflc.calcValue(1 + osim_pflc_args[1]) - 1) < 1e6  # verify that passive-force-length curve behaves as expected
            env.sim.model.actuator_gainprm[actuator_id, 7] = osim_pflc.calcValue(env.sim.model.actuator_gainprm[actuator_id, 5])  #fpmax
            env.sim.model.actuator_gainprm[actuator_id, 8] = float(osim_file.max_eccentric_velocity_force_multiplier[actuator_fvc_name])  #fvmax
    else:
        #raise NotImplementedError
        pass

    # MUJOCO MODEL ADJUSTMENT: adjust activation/deactivation time constants
    env.sim.model.actuator_dynprm[actuator_indexlist, :2] = compute_time_activation_constants(osim_file)

    # MUJOCO MODEL ADJUSTMENT: adjust actuator length ranges
    # -> minimize error between MuJoCo and OpenSim optimal fiber length w.r.t. actuator_lengthrange
    LO_osim = np.array(
        [float(osim_file.optimal_fiber_length[env.sim.model.actuator_id2name(actuator_id)]) for actuator_id
         in actuator_indexlist])
    LT_osim = np.array(
        [float(osim_file.tendon_slack_length[env.sim.model.actuator_id2name(actuator_id)]) for actuator_id
         in actuator_indexlist])
    optPennationAngle_osim = np.array(
        [float(osim_file.pennation_angle_at_optimal[env.sim.model.actuator_id2name(actuator_id)])
         for actuator_id in actuator_indexlist])
    sol = minimize(mujoco_LO_loss, env.sim.model.actuator_lengthrange[actuator_indexlist],
                   args=(env, LO_osim, LT_osim, optPennationAngle_osim, actuator_indexlist))
    assert sol.success, "Computation of actuator length ranges was not successful."
    env.sim.model.actuator_lengthrange[actuator_indexlist] = sol.x.reshape((len(actuator_indexlist), 2))


def get_mujoco_body(mujoco_xml, bodyname):
    """
    Returns parsed dict of a given MuJoCo body.
    :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
    :param bodyname: name of MuJoCo body (str)
    :return: parsed dict of MuJoCo body with name 'bodyname'
    """

    if type(mujoco_xml) == str:
        filepath = mujoco_xml
        with open(filepath, "r") as f:
            text = f.read()
        mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    main_body = [i for i in mujoco_xml['mujoco']['worldbody']['body'] if i['@name'] == "thorax"]
    assert len(main_body) == 1, "ERROR: More than one body with name 'thorax' detected!"
    current_body = main_body[0]

    while ('body' in current_body) and (current_body['@name'] != bodyname):
        current_body = current_body['body']
    assert (current_body['@name'] == bodyname), f"Body {bodyname} was not found!"

    return current_body


def get_mujoco_geom(mujoco_xml, geomname):
    """
    Returns parsed dict of a given MuJoCo geom.
    :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
    :param bodyname: name of MuJoCo geom (str)
    :return: parsed dict of MuJoCo geom with name 'geomname'
    """

    if type(mujoco_xml) == str:
        filepath = mujoco_xml
        with open(filepath, "r") as f:
            text = f.read()
        mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    main_body = [i for i in mujoco_xml['mujoco']['worldbody']['body'] if i['@name'] == "thorax"]
    assert len(main_body) == 1, "ERROR: More than one body with name 'thorax' detected!"
    current_body = main_body[0]
    current_body_geoms = current_body['geom'] if (
                ('geom' in current_body) and (type(current_body['geom']) == list)) else [current_body['geom']] if (
                ('geom' in current_body) and (type(current_body['geom']) == dict)) else []

    while ('body' in current_body) and all([i['@name'] != geomname for i in current_body_geoms]):
        current_body = current_body['body']
        current_body_geoms = current_body['geom'] if (
                    ('geom' in current_body) and (type(current_body['geom']) == list)) else [current_body['geom']] if (
                    ('geom' in current_body) and (type(current_body['geom']) == dict)) else []
    for current_body_geom in current_body_geoms:
        if current_body_geom['@name'] == geomname:
            return current_body_geom

    raise AssertionError(f"Geom {geomname} was not found!")
    #print(f"Geom {geomname} was not found!")  # None is returned


def find_body_of_site(mujoco_xml, sitename):
    """
    Finds parent body of a given site in MuJoCo.
    :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
    :param bodyname: name of MuJoCo site (str)
    :return: name of parent body (str), if site exists (otherwise None)
    """

    if type(mujoco_xml) == str:
        filepath = mujoco_xml
        with open(filepath, "r") as f:
            text = f.read()
        mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    main_body = [i for i in mujoco_xml['mujoco']['worldbody']['body'] if i['@name'] == "thorax"]
    assert len(main_body) == 1, "ERROR: More than one body with name 'thorax' detected!"
    current_body = main_body[0]
    current_body_sites = current_body['site'] if (
                ('site' in current_body) and (type(current_body['site']) == list)) else [current_body['site']] if (
                ('site' in current_body) and (type(current_body['site']) == dict)) else []

    while ('body' in current_body) and all([i['@name'] != sitename for i in current_body_sites]):
        current_body = current_body['body']
        current_body_sites = current_body['site'] if 'site' in current_body else []
    if any([i['@name'] == sitename for i in current_body_sites]):
        return current_body['@name']


def append_OpenSim_wrap_object_to_MuJoCo_body(current_body, wrap_object, object_type="WrapCylinder"):
    """
    Appends a given tendon wrapping object to a MuJoCo body.
    :param current_body: parsed dict of the MuJoCo body the tendon wrapping object should be attached to
    :param wrap_object: OpenSim tendon wrapping object (OrderedDict,
                        typically included in an instance of opensim_file such as opensim_file.WrapCylinder)
    :param object_type: type of tendon wrapping object (str)
    :return: name of wrap_object, if appending to MuJoCo model failed (otherwise None)
    """

    ELLIPSOID_THRESHOLD = 3  #used below to decide whether to approximate an ellipsoid via sphere or cylinder

    if 'geom' not in current_body:
        current_body['geom'] = []
    elif type(current_body['geom']) == dict:
        current_body['geom'] = [current_body['geom']]

    if 'site' not in current_body:
        current_body['site'] = []
    elif type(current_body['site']) == dict:
        current_body['site'] = [current_body['site']]

    if object_type == "WrapCylinder":
        current_body['geom'].append({'@name': wrap_object['@name'],
                                     '@type': 'cylinder',
                                     '@class': 'WrapCylinder',
                                     '@size': f"{wrap_object['radius']} {0.5 * float(wrap_object['length'])}",
                                     # MuJoCo attribute: half-height; OpenSim attribute: height/length (but might be unused by MuJoCo, since tendon geometries must be spheres or INFINITE cylinders)
                                     '@euler': wrap_object['xyz_body_rotation'],
                                     '@pos': wrap_object['translation']})

        if wrap_object['quadrant'] != "all":
            # add tendon side site on the correct side outside of the geometry (here, 2*radius is added to center):
            sidesite_offset = 2 * float(wrap_object['radius']) * np.array(
                ["x" in wrap_object['quadrant'], "y" in wrap_object['quadrant'], "z" in wrap_object['quadrant']])
            assert sum(sidesite_offset) == 2 * float(
                wrap_object['radius'])  # only one side should be specified in wrap_object['quadrant']
            if "-" in wrap_object['quadrant']:  # site is on opposite direction of geometry
                sidesite_offset *= -1
            current_body['site'].append({'@name': wrap_object['@name'] + '-sidesite',
                                         '@pos': array_to_strinterval(
                                             strinterval_to_nparray(wrap_object['translation']) + sidesite_offset)})
    elif object_type == "WrapSphere":
        current_body['geom'].append({'@name': wrap_object['@name'],
                                     '@type': 'sphere',
                                     '@class': 'WrapSphere',
                                     '@size': wrap_object['radius'],
                                     '@euler': wrap_object['xyz_body_rotation'],
                                     '@pos': wrap_object['translation']})

        if wrap_object['quadrant'] != "all":
            # add tendon side site on the correct side outside of the geometry (here, 2*radius is added to center):
            sidesite_offset = 2 * float(wrap_object['radius']) * np.array(
                ["x" in wrap_object['quadrant'], "y" in wrap_object['quadrant'], "z" in wrap_object['quadrant']])
            assert sum(sidesite_offset) == 2 * float(
                wrap_object['radius'])  # only one side should be specified in wrap_object['quadrant']
            if "-" in wrap_object['quadrant']:  # site is on opposite direction of geometry
                sidesite_offset *= -1
            current_body['site'].append({'@name': wrap_object['@name'] + '-sidesite',
                                         '@pos': array_to_strinterval(
                                             strinterval_to_nparray(wrap_object['translation']) + sidesite_offset)})
    elif object_type == "WrapEllipsoid":
        ellipsoid_radii = sorted(strinterval_to_nparray(wrap_object['dimensions']))
        assert len(ellipsoid_radii) == 3
        if (ellipsoid_radii[-1] / ellipsoid_radii[-2]) > ELLIPSOID_THRESHOLD:
            assert (ellipsoid_radii[-2] / ellipsoid_radii[
                -3]) <= ELLIPSOID_THRESHOLD, "Tendon Wrapping Error: Smaller two radii of ellipsoid should not differ that much..."
            if np.argmax(ellipsoid_radii) == 0:
                #TODO: verify these rotations!
                # rotate axis around y-axis such that the cylinder is orientied along the z-axis (adjust wrap_object['xyz_body_rotation'] and wrap_object['quadrant'])
                xyz_body_rotation = Rotation.from_matrix(Rotation.from_euler('Y', 90, degrees=True).as_matrix() @
                                        Rotation.from_euler(seq="XYZ", angles=wrap_object['xyz_body_rotation'], degrees=False).as_matrix()).as_euler('XYZ', degrees=False)

                translation = Rotation.from_euler('Y', 90, degrees=True).apply(wrap_object['translation'])
                quadrant_mapping = {'x': 'z', 'y': 'y', 'z': '-x', '-x': '-z', '-y': '-y', '-z': 'x', 'all': 'all'}
                quadrant = quadrant_mapping[wrap_object['quadrant']]
            elif np.argmax(ellipsoid_radii) == 1:
                #TODO: verify these rotations!
                # rotate axis around x-axis such that the cylinder is orientied along the z-axis (adjust wrap_object['xyz_body_rotation'] and wrap_object['quadrant'])
                xyz_body_rotation = Rotation.from_matrix(Rotation.from_euler('X', -90, degrees=True).as_matrix() @
                                        Rotation.from_euler(seq="XYZ", angles=wrap_object['xyz_body_rotation'], degrees=False).as_matrix()).as_euler('XYZ', degrees=False)

                translation = Rotation.from_euler('X', 90, degrees=True).apply(wrap_object['translation'])
                quadrant_mapping = {'x': 'x', 'y': '-z', 'z': 'y', '-x': '-x', '-y': 'z', '-z': '-y', 'all': 'all'}
                quadrant = quadrant_mapping[wrap_object['quadrant']]
            else:
                xyz_body_rotation = wrap_object['xyz_body_rotation']
                translation = wrap_object['translation']
                quadrant = wrap_object['quadrant']
            # use cylinder as approximation of ellipsoid
            cylinder_radius = np.mean(ellipsoid_radii[:2])  # use mean radius of smallest two radii
            assert 'z' not in quadrant, f"ERROR: Side site of {wrap_object['@name']} cannot be placed along z-axis of (infinite) cylinder!"
            current_body['geom'].append({'@name': wrap_object['@name'],
                                         '@type': 'cylinder',
                                         '@class': 'WrapEllipsoid',
                                         '@size': f"{cylinder_radius} {0.5 * float(ellipsoid_radii[2])}",
                                         # MuJoCo attribute: half-height; OpenSim attribute: height/length (but might be unused by MuJoCo, since tendon geometries must be spheres or INFINITE cylinders)
                                         '@euler': xyz_body_rotation,
                                         '@pos': translation})

            if quadrant != "all":
                # add tendon side site on the correct side outside of the geometry (here, 2*radius is added to center):
                sidesite_offset = 2 * cylinder_radius * np.array(["x" in quadrant, "y" in quadrant, "z" in quadrant])
                assert sum(sidesite_offset) == 2 * cylinder_radius  # only one side should be specified in wrap_object['quadrant']
                if "-" in quadrant:  # site is on opposite direction of geometry
                    sidesite_offset *= -1
                current_body['site'].append({'@name': wrap_object['@name'] + '-sidesite',
                                             '@pos': array_to_strinterval(strinterval_to_nparray(
                                                 translation) + sidesite_offset)})
        else:
            # use sphere as approximation of ellipsoid
            assert wrap_object['quadrant'] in ['x', 'y', 'z', '-x', '-y', '-z', 'all'], f"Unknown quadrant value ({wrap_object['quadrant']}) for {wrap_object['@name']}."
            if wrap_object['quadrant'] != "all":
                sphere_radius = strinterval_to_nparray(wrap_object['dimensions'])[
                    ['x', 'y', 'z'].index(wrap_object['quadrant'][-1])]  # use radius in direction of side site
            else:
                sphere_radius = np.mean(strinterval_to_nparray(wrap_object['dimensions']))  # use mean radius
            current_body['geom'].append({'@name': wrap_object['@name'],
                                         '@type': 'sphere',
                                         '@class': 'WrapEllipsoid',
                                         '@size': sphere_radius,
                                         '@euler': wrap_object['xyz_body_rotation'],
                                         '@pos': wrap_object['translation']})

            if wrap_object['quadrant'] != "all":
                # add tendon side site on the correct side outside of the geometry (here, 2*radius is added to center):
                sidesite_offset = 2 * sphere_radius * np.array(
                    ["x" in wrap_object['quadrant'], "y" in wrap_object['quadrant'],
                     "z" in wrap_object['quadrant']])
                assert sum(
                    sidesite_offset) == 2 * sphere_radius  # only one side should be specified in wrap_object['quadrant']
                if "-" in wrap_object['quadrant']:  # site is on opposite direction of geometry
                    sidesite_offset *= -1
                current_body['site'].append({'@name': wrap_object['@name'] + '-sidesite',
                                             '@pos': array_to_strinterval(strinterval_to_nparray(
                                                 wrap_object['translation']) + sidesite_offset)})
    elif object_type == "WrapTorus":
        current_body['geom'].append({'@name': wrap_object['@name'],
                                     '@type': 'sphere',
                                     '@class': 'WrapTorus',
                                     '@size': wrap_object['inner_radius'],
                                     # ignores outer_radius of torus geometry, which in theory might act as an obstacle for (other) tendon paths
                                     '@euler': wrap_object['xyz_body_rotation'],
                                     '@pos': wrap_object['translation']})

        # add tendon side site inside the sphere to approximate OpenSim torus wrapping:
        current_body['site'].append({'@name': wrap_object['@name'] + '-sidesite_torus',
                                     '@pos': wrap_object['translation']})
    else:
        # raise TypeError(f"Unknown OpenSim wrap object type '{object_type}'.")
        print(
            f"WARNING: Wrap object {wrap_object['@name']} has not been added to MuJoCo model, since its object type '{object_type}' is unknown!")
        return wrap_object['@name']


def append_OpenSim_wrap_path_to_MuJoCo_model(mujoco_xml, tendon_name, wrap_path):
    """
    References a tendon wrapping object (referenced in OpenSim wrap_path) within a tendon spatial path.
    WARNING: MuJoCo requires two tendon wrapping objects to be separated by a site. If multiple wrapping objects are
     to be placed between the same two sites, only the geometry with largest size is used.
    :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
    :param tendon_name: name of tendon to whose spatial path the geometries within wrap_path are to be added (str)
    :param wrap_path: reference within OpenSim tendon wrapping path (OrderedDict,
                        typically included in opensim_file.PathWrap[tendon_name])
    :return: None
    """

    if type(mujoco_xml) == str:
        filepath = mujoco_xml
        with open(filepath, "r") as f:
            text = f.read()
        mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    spatial_tendons = mujoco_xml['mujoco']['tendon']['spatial']
    if type(spatial_tendons) == dict:
        spatial_tendons = [spatial_tendons]

    # tendon_found = False
    # geom_brackets_counter = 0
    for spatial_tendon in spatial_tendons:  # find corresponding 'spatial' tag in tendon object
        if spatial_tendon['@name'] == tendon_name:
            if 'geom' not in spatial_tendon:
                spatial_tendon['geom'] = []
            elif type(spatial_tendon['geom']) == dict:
                spatial_tendon['geom'] = [spatial_tendon['geom']]

            sidesite_exists = find_body_of_site(mujoco_xml, wrap_path['wrap_object'] + '-sidesite')
            sidesite_torus_exists = find_body_of_site(mujoco_xml, wrap_path['wrap_object'] + '-sidesite_torus')

            wrap_path_range = (strinterval_to_nparray(wrap_path['range'])).astype(np.int64)
            if wrap_path_range[0] == -1:  # corresponds to "from first site..." in OpenSim
                wrap_path_range[0] = 1
            if wrap_path_range[1] == -1:  # corresponds to "...to last site" in OpenSim
                wrap_path_range[1] = len(spatial_tendon['site'])

            current_geom_name = wrap_path['wrap_object']
            current_geom = get_mujoco_geom(mujoco_xml, current_geom_name)
            current_geom_size = float(current_geom['@size']) if current_geom['@type'] == 'sphere' else \
            strinterval_to_nparray(current_geom['@size'])[0]  # only spheres and cylinders are allowed for tendon geoms

            # place geometry between first site included in OpenSim range and its direct successor
            try:
                assert wrap_path_range[1] <= len(spatial_tendon['site']), f"Error in range of {wrap_path['wrap_object']} object."
            except AssertionError:
                if wrap_path_range[1] > len(spatial_tendon['site']):
                    old_wrap_path_range = wrap_path_range.copy()
                    wrap_path_range[1] = len(spatial_tendon['site'])
                    print(f"WARNING: Error in range of {wrap_path['wrap_object']} object. Replacing {old_wrap_path_range} by {wrap_path_range}.\n")
            site_range_indices = list(
                range(wrap_path_range[0] - 1, wrap_path_range[1] - 1))  # [int(i) - 1 for i in wrap_path_range]
            for site_range_index in site_range_indices:
                new_order = spatial_tendon['site'][site_range_index]['@__order__'] + 0.1
                geom_dict = {'@geom': wrap_path['wrap_object']}

                # if sidesites are available, add them to tendon geometry object:
                if sidesite_exists:
                    geom_dict['@sidesite'] = wrap_path['wrap_object'] + '-sidesite'
                elif sidesite_torus_exists:
                    geom_dict['@sidesite'] = wrap_path['wrap_object'] + '-sidesite_torus'

                # check if there is already some geometry placed directly after current site:
                other_geoms_names = [i['@geom'] for i in spatial_tendon['geom'] if i['@__order__'] == new_order]
                if len(other_geoms_names) > 0:
                    for other_geoms_name in other_geoms_names:  # 'other_geoms_names' should only include one element...
                        other_geom = get_mujoco_geom(mujoco_xml, other_geoms_name)
                        other_geom_size = float(other_geom['@size']) if other_geom['@type'] == 'sphere' else \
                        strinterval_to_nparray(other_geom['@size'])[
                            0]  # only spheres and cylinders are allowed for tendon geoms

                        # input(((current_geom_name, current_geom_size), (other_geoms_name, other_geom_size)))
                        if current_geom_size > other_geom_size:
                            geom_dict['@__order__'] = new_order

                            # remove other tendon geometry from this position in tendon path
                            other_geom_tendonpath = [i for i in spatial_tendon['geom'] if
                                                     (i['@geom'] == other_geoms_name) and (
                                                                 i['@__order__'] == new_order)]
                            assert len(other_geom_tendonpath) == 1
                            other_geom_tendonpath = other_geom_tendonpath[0]
                            spatial_tendon['geom'].remove(other_geom_tendonpath)
                        else:  # do not add new tendon geom here, as its size is too small compared to existing tendon geom
                            break
                    else:
                        geom_dict['@__order__'] = new_order
                        spatial_tendon['geom'].append(geom_dict)
                else:
                    geom_dict['@__order__'] = new_order
                    spatial_tendon['geom'].append(geom_dict)

            # input((spatial_tendon['site'], spatial_tendon['geom']))
            #             geom_dict = {'@geom': wrap_path['wrap_object']}
            #             current_tendon_sites = spatial_tendon['site'].copy()

            #             # if no site is placed immediately after this geom, use site before and site after this geom as "brackets" and re-insert all three objects to tendon path (see https://mujoco.readthedocs.io/en/latest/XMLreference.html#tendon-spatial and https://mujoco.readthedocs.io/en/latest/_static/tendon.xml)
            #             next_site_orderindex = min([(i['@__order__']) if i['@__order__'] > geom_dict['@__order__'] else np.inf for i in spatial_tendon['site']])
            #             if spatial_tendon['geom'] == []:
            #                 next_geom_orderindex = np.inf
            #             else:
            #                 next_geom_orderindex = min([(i['@__order__']) if ('@__order__' in i) and (i['@__order__'] >= geom_dict['@__order__']) else np.inf for i in spatial_tendon['geom']])
            #             if next_site_orderindex >= next_geom_orderindex:
            #                 geom_brackets_counter += 1
            #                 tendon_site_before_geom = current_tendon_sites[site_range_indices[0]].copy()
            #                 tendon_site_before_geom['@__order__'] = next_geom_orderindex + (0.1 ** geom_brackets_counter)
            #                 spatial_tendon['site'].append(tendon_site_before_geom)

            #                 geom_dict['@__order__'] = next_geom_orderindex + 2 * (0.1 ** geom_brackets_counter)

            #                 tendon_site_after_geom = current_tendon_sites[site_range_indices[0] + 1].copy()
            #                 tendon_site_after_geom['@__order__'] = next_geom_orderindex + 3 * (0.1 ** geom_brackets_counter)
            #                 spatial_tendon['site'].append(tendon_site_after_geom)
            #                 input((spatial_tendon, next_geom_orderindex))

            # tendon_found = True
            break
    # assert tendon_found, f"Tendon '{tendon_name}' was not found in MuJoCo model."

    if len(spatial_tendons) == 1:
        mujoco_xml['mujoco']['tendon']['spatial'] = spatial_tendons[0]


def _append_OpenSim_path_point_to_MuJoCo_model(mujoco_xml, tendon_name, path_point):
    """
    References a path point object (referenced in OpenSim path_point) within a tendon spatial path.
    WARNING: Since MuJoCo only allows for static path points, both ConditionalPathPoint and MovingPathPoint objects
     need to be transferred to regular PathPoint objects.
    :param mujoco_xml: MuJoCo model (filepath or parsed dict obtained from xmltodict.parse())
    :param tendon_name: name of tendon to whose spatial path the geometries within wrap_path are to be added (str)
    :param path_point: reference within OpenSim tendon path point (OrderedDict,
                        typically included in opensim_file.PathPoint[tendon_name])
    :return: None
    """

    if type(mujoco_xml) == str:
        filepath = mujoco_xml
        with open(filepath, "r") as f:
            text = f.read()
        mujoco_xml = xmltodict.parse(text, dict_constructor=dict, process_namespaces=True, ordered_mixed_children=True)

    spatial_tendons = mujoco_xml['mujoco']['tendon']['spatial']
    if type(spatial_tendons) == dict:
        spatial_tendons = [spatial_tendons]

    for spatial_tendon in spatial_tendons:  # find corresponding 'spatial' tag in tendon object
        if spatial_tendon['@name'] == tendon_name:
            if 'site' not in spatial_tendon:
                spatial_tendon['site'] = []
            elif type(spatial_tendon['site']) == dict:
                spatial_tendon['site'] = [spatial_tendon['site']]

            site_dict = {'@site': path_point['@name']}
            if site_dict not in spatial_tendon['site']:
                spatial_tendon['site'].append(site_dict)
            else:
                pass
            break

    if len(spatial_tendons) == 1:
        mujoco_xml['mujoco']['tendon']['spatial'] = spatial_tendons[0]


def compute_time_activation_constants(osim_file_or_path, act_linearization=0.5, ctrl_linearization=0.5):
    """
    Compute MuJoCo time activation/deactivation constants from OpenSim model with Schuette1993 muscles.
    IMPORTANT: Assumes that same activation dynamics apply to all muscle actuators! (If this is not the case,
    code below could easily be modified to compute constants for each actuator separately.)

    :param osim_file_or_path: OpenSim file path or parsed class
    :param act_linearization: actuation value to linearize at
    :param ctrl_linearization: control value to linearize at
    :return: MuJoCo time activation constant, MuJoCo time deactivation constant
    """
    if type(osim_file_or_path) == str:
        osim_file = opensim_file(osim_file_or_path)
    elif type(osim_file_or_path) == opensim_file:
        osim_file = osim_file_or_path
    else:
        raise TypeError(
            "Argument of compute_time_activation_constants() needs to be OpenSim file path or parsed class.")

    if osim_file.Schutte1993Muscle_Deprecated:  #if "Schutte1993Muscle_Deprecated" muscles are used
        activation1 = list(set(osim_file.activation1.values()))
        assert len(activation1) == 1
        activation1 = float(activation1[0])

        activation2 = list(set(osim_file.activation2.values()))
        assert len(activation2) == 1
        activation2 = float(activation2[0])

        time_scale = list(set(osim_file.time_scale.values()))
        assert len(time_scale) == 1
        time_scale = float(time_scale[0])

        time_act = time_scale / ((0.5 + 1.5 * act_linearization) * (activation1 * ctrl_linearization + activation2))
        time_deact = time_scale * (0.5 + 1.5 * act_linearization) / activation2

        return time_act, time_deact
    elif osim_file.Millard2012EquilibriumMuscle or osim_file.Thelen2003Muscle:  #if "Millard2012EquilibriumMuscle" or "Thelen2003Muscle" muscles are used
        if osim_file.activation_time_constant:
            activation_time_constant = list(set(osim_file.activation_time_constant.values()))
            assert len(activation_time_constant) == 1
            activation_time_constant = float(activation_time_constant[0])
        else:
            activation_time_constant = 0.01 if osim_file.Millard2012EquilibriumMuscle else 0.015 #OpenSim default

        if osim_file.deactivation_time_constant:
            deactivation_time_constant = list(set(osim_file.deactivation_time_constant.values()))
            assert len(deactivation_time_constant) == 1
            deactivation_time_constant = float(deactivation_time_constant[0])
        else:
            deactivation_time_constant = 0.04 if osim_file.Millard2012EquilibriumMuscle else 0.05 #OpenSim default

        return activation_time_constant, deactivation_time_constant
    else:
        raise TypeError("OpenSim model includes muscle objects of unknown type.")


def mujoco_get_LO(env, actuator_id_or_name):
    """
    Returns optimal fiber length LO of a given actuator in MuJoCo.
    """

    if type(actuator_id_or_name) == str:
        actuator_id = env.sim.model.actuator_name2id(actuator_id_or_name)
    else:
        actuator_id = actuator_id_or_name
    return (env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_lengthrange[actuator_id][
        1]) / (env.sim.model.actuator_gainprm[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][1])


def mujoco_get_LT(env, actuator_id_or_name):
    """
    Returns constant tendon length LT of a given actuator in MuJoCo.
    """

    if type(actuator_id_or_name) == str:
        actuator_id = env.sim.model.actuator_name2id(actuator_id_or_name)
    else:
        actuator_id = actuator_id_or_name
    return env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][
        0] * mujoco_get_LO(env, actuator_id)


def mujoco_LO_loss(actuator_lengthranges, env, LO_osim, LT_osim, optPennationAngle_osim, actuator_indexlist,
            use_optPennationAngle=True):
    """
    Computes squared Euclidean distance between MuJoCo and OpenSim model,
    regarding both optimal fiber length and constant tendon length/tendon slack length for all muscle actuators,
    given an array of MuJoCo actuator length ranges (i.e., ranges of the complete musculotendon lengths).

    :param actuator_lengthranges: array of MuJoCo tendon length (=complete actuator length) ranges
    :param env: Gym environment of MuJoCo model
    :param LO_osim: array of OpenSim optimal fiber lengths
    :param LT_osim: array of OpenSim tendon slack lengths (or any reasonable constant tendon lengths)
    :param optPennationAngle_osim: array of OpenSim pennation angles at optimum
            (i.e., angle between tendon and fibers at optimal fiber length expressed in radians)
    :param actuator_indexlist: list of MuJoCo actuator indices that correspond to muscle actuators
    :param use_optPennationAngle: Boolean; if this set to True, MuJoCo optimal fiber lengths LO should match
            OpenSim optimal fiber lengths LO_osim * cos(OpenSim pennation angle at optimum); otherwise, LO should match LO_osim
    :return: squared (unweighted) Euclidean distance of optimal fiber lengths and constant tendon lengths between MuJoCo and OpenSim
    """
    env.sim.model.actuator_lengthrange[actuator_indexlist] = actuator_lengthranges.reshape(
        (len(actuator_indexlist), 2))

    LO = np.array([mujoco_get_LO(env, actuator_id) for actuator_id in actuator_indexlist])
    LT = np.array([mujoco_get_LT(env, actuator_id) for actuator_id in actuator_indexlist])

    pennationAngleHelper = optPennationAngle_osim if use_optPennationAngle else np.zeros(LO_osim.shape)
    return np.linalg.norm(LO - LO_osim * np.cos(pennationAngleHelper)) ** 2 + np.linalg.norm(LT - LT_osim) ** 2


def print_musculotendon_properties(env, osim_file, stdout_file=None):
    """
    Prints and compares musculotendon properties of MuJoCo model and OpenSim model.
    :param env: Gym environment of MuJoCo model
    :param osim_file: OpenSim model (filepath or parsed opensim_file class)
    :param stdout_file: file the output stream is redirected to
    :return: None
    """

    if stdout_file:
        orig_stdout = sys.stdout
        f = open(stdout_file, 'w')
        sys.stdout = f

    if type(osim_file) == str:
        osim_file = opensim_file(osim_file)

    scale_ratio = np.unique(env.sim.model.actuator_gainprm[:, :2])
    assert scale_ratio.shape == (2,), "Scale ratios should be the same for all actuators."
    print(f"Scale ratios of optimal fiber length: {scale_ratio}")

    for actuator_id in [i for i in range(env.sim.model.nu) if
                        env.sim.model.actuator_trntype[i] == 3]:  # only consider tendon actuators
        ## (for details, see https://mujoco.readthedocs.io/en/latest/modeling.html#muscle-actuators)
        LO = (env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_lengthrange[actuator_id][
            1]) / (
                     env.sim.model.actuator_gainprm[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][1])
        LT = env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][0] * LO
        # LT = env.sim.model.actuator_lengthrange[actuator_id][1] - env.sim.model.actuator_gainprm[actuator_id][1] * LO
        FO = env.sim.model.actuator_gainprm[actuator_id][2]

        actuator_name = env.sim.model.actuator_id2name(actuator_id)
        tendon_id = env.sim.model.actuator_trnid[actuator_id][0]
        print(f"{actuator_name}:\n\tactuator_length (MuJoCo): {env.sim.data.actuator_length[actuator_id]}"
              f"\n\tactuator length range (MuJoCo): {env.sim.model.actuator_lengthrange[actuator_id]}"
              f"\n\tmuscle length LM (MuJoCo): {env.sim.data.actuator_length[actuator_id] - LT}"
              f"\n\n\tconstant tendon length LT (MuJoCo): {LT}")
        if actuator_name in osim_file.tendon_slack_length:
            print(f"\ttendon slack length (OpenSim): {osim_file.tendon_slack_length[actuator_name]}")
        # print(f"{actuator_name}:\n\tactuator_length (MuJoCo): {env.sim.data.actuator_length[actuator_id]}\n\ttendon length (MuJoCo): {env.sim.data.ten_length[tendon_id]}\n\tmuscle length (MuJoCo): {env.sim.data.actuator_length[actuator_id] - env.sim.data.ten_length[tendon_id]}")
        # print(f"{actuator_name}:actuator_length: \n\t{env.sim.data.actuator_length}ten_length: \n\t{env.sim.model.tendon_length0}")

        print(f"\n\toptimal fiber length LO (MuJoCo): {LO}")
        if actuator_name in osim_file.optimal_fiber_length:
            print(
                f"\toptimal fiber length [w/opt. pennation angle] (OpenSim): {float(osim_file.optimal_fiber_length[actuator_name]) * np.cos(float(osim_file.pennation_angle_at_optimal[actuator_name]))}")
        print(f"\n\tmaximum isometric force FO (MuJoCo): {FO}")
        if actuator_name in osim_file.max_isometric_force:
            print(f"\tmaximum isometric force (OpenSim): {osim_file.max_isometric_force[actuator_name]}")

    print(f"\nMuJoCo time activation/deactivation constants: {compute_time_activation_constants(osim_file)}")

    if stdout_file:
        sys.stdout = orig_stdout
        f.close()


def strinterval_to_nparray(interval):
  """
  Converts string interval representation to numpy array consisting of floats.
  """

  insplit = interval.split(" ")
  return np.array([float(inval) for inval in insplit], dtype=float)


def array_to_strinterval(array):
  """
  Converts array consisting of floats to string interval representation with whitespace between two numbers.
  """

  return ' '.join(['%8g' % num for num in array])


class OsimModelEnv(object):
    """
    Source: https://github.com/stanfordnmbl/osim-rl (slightly adapted)

    OpenSim interface
    The amin purpose of this class is to provide wrap all
    the necessery elements of OpenSim in one place
    The actual RL environment then only needs to:
    - open a model
    - actuate
    - integrate
    - read the high level description of the state
    The objective, stop condition, and other gym-related
    methods are enclosed in the OsimEnv class
    """

    # Initialize simulation
    model = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0

    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    joint_values_istep = None
    prev_joint_values = None
    joint_values = None

    muscle_values_istep = None
    prev_muscle_values = None
    muscle_values = None

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize=False, integrator_accuracy=5e-5, stepsize=0.01):
        self.integrator_accuracy = integrator_accuracy
        self.stepsize = stepsize
        self.model = osim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = osim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle
        for j in range(self.muscleSet.getSize()):
            func = osim.Constant(1.0)
            self.brain.addActuator(self.muscleSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        self.noutput = self.muscleSet.getSize()

        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

        self.timestep_limit = np.inf

        self.action_space = spaces.Box(np.array([0.0] * self.get_action_space_size()), np.array([1.0] * self.get_action_space_size()))


    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i, self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i, self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i, self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i, self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i, self.markerSet.get(i).getName())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        action = np.clip(np.array(action), 0.0, 1.0)
        self.last_action = action

        brain = osim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = osim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(float(action[j]))

    """
    Directly modifies activations in the current state.
    """

    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])
        self.reset_manager()

    """
    Get activations in the given state.
    """

    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in
                                      range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in
                                      range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in
                                      range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]

            res["body_pos_rot"][name] = [
                body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html

        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def compute_muscle_values(self):
        res_osim = {}
        for i in range(self.muscleSet.getSize()):
            muscle = osim.Millard2012EquilibriumMuscle.safeDownCast(self.muscleSet.get(i))
            name = muscle.getName()
            res_osim[name] = (muscle.getActivation(self.state), muscle.getActivationDerivative(self.state), muscle.getFiberLength(self.state))
        return res_osim

    def get_muscle_values(self):
        if self.muscle_values_istep != self.istep:
            self.prev_muscle_values = self.muscle_values
            self.muscle_values = self.compute_muscle_values()
            self.muscle_values_istep = self.istep
        return self.muscle_values

    def compute_joint_values(self):
        res_osim = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            for j in range(joint.numCoordinates()):
                coordinate = joint.get_coordinates(j)
                coordinate_name = coordinate.getSpeedName().split("/")[0]
                res_osim[coordinate_name] = (coordinate.getValue(self.state), coordinate.getSpeedValue(self.state)) #, coordinate.getAccelerationValue(self.state))
        return res_osim

    def get_joint_values(self):
        if self.joint_values_istep != self.istep:
            self.prev_joint_values = self.joint_values
            self.joint_values = self.compute_joint_values()
            self.joint_values_isteps = self.istep
        return self.joint_values

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def _reset(self):  #'reset' from osim-rl OsimModel class
        self.state = self.model.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

    def reset(self, project=True, obs_as_dict=True):  #from osim-rl OsimEnv class
        self._reset()

        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_state_desc()  #self.get_observation_dict()
        return self.get_state_desc()  #self.get_observation()

    def step(self, action, project=True, obs_as_dict=True):
        self.prev_state_desc = self.get_state_desc()
        self.actuate(action)
        self.integrate()

        if project:
            if obs_as_dict:
                obs = self.get_state_desc()  #self.get_observation_dict()
            else:
                obs = self.get_state_desc()  #self.get_observation()
        else:
            obs = self.get_state_desc()  #self.get_state_desc()

        return [obs, self.get_reward(), self.is_done() or (self.istep >= self.timestep_limit), {}]

    def get_state(self):
        return osim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.istep = int(self.state.getTime() / self.stepsize)  # TODO: remove istep altogether
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        self.state = self.manager.integrate(self.stepsize * self.istep)

    def is_done(self):
        #raise NotImplementedError
        return False

    def get_reward(self):
        #raise NotImplementedError
        return 0


def osim_get_coordinate(osim_env, coordinate_name, joint_name=None):
    """
    Get OpenSim Coordinate by name.
    :param osim_env: OsimModelEnv class of OpenSim model
    :param coordinate_name: name of OpenSim coordinate (str)
    :param joint_name: name of OpenSim joint (str), if available; by default, all joints are scanned
    :return: OpenSim Coordinate object (opensim.simulation.Coordinate)
    """
    if joint_name is not None:
        coordinate = [coordinate for i in range(osim_env.jointSet.getSize()) if ((joint := osim_env.jointSet.get(i)).getName() == joint_name) for j in range(joint.numCoordinates()) if (coordinate := joint.get_coordinates(j)).getName() == coordinate_name]
    else:
        coordinate = [coordinate for i in range(osim_env.jointSet.getSize()) if (joint := osim_env.jointSet.get(i)) is not None for j in range(joint.numCoordinates()) if (coordinate := joint.get_coordinates(j)).getName() == coordinate_name]
    assert len(coordinate) >= 1, f"ERROR: Coordinate '{coordinate_name}' not found!"
    assert len(coordinate) == 1, f"ERROR: There exists more than one coordinate with name '{coordinate_name}'!"
    return coordinate[0]


def check_model_state_equality(mujoco_env, osim_env):
    """
    Checks whether MuJoCo and OpenSim model are in same state.
    :param mujoco_env: Gym environment of MuJoCo model
    :param osim_env: (customized) OsimModelEnv class of OpenSim model
    :return: True, if the position, speed, and acceleration values of all common joints match,
    as well as muscle and body states, else False
    """
    ## Joint state
    res_mujoco = mujoco_env.get_joint_values()
    res_osim = osim_env.get_joint_values()

    mujoco_only_joints = {i for i in res_mujoco.keys() if i not in res_osim.keys()}
    osim_only_joints = {i for i in res_osim.keys() if i not in res_mujoco.keys()}
    print(f"WARNING: Joints only available in MuJoCo model: {mujoco_only_joints}")
    print(f"WARNING: Joints only available in OpenSim model: {osim_only_joints}")

    res_mujoco_common = {k: v for k, v in res_mujoco.items() if k not in mujoco_only_joints}
    res_osim_common = {k: v for k, v in res_osim.items() if k not in osim_only_joints}
    # input(yaml.dump(res_osim, default_flow_style=False))
    if res_mujoco_common != res_osim_common:
        print("WARNING: Joint states do not match between MuJoCo and OpenSim!")

    ## Muscle state
    res_mujoco_muscles = mujoco_env.get_muscle_values()
    res_osim_muscles = osim_env.get_muscle_values()

    mujoco_only_muscles = {i for i in res_mujoco_muscles.keys() if i not in res_osim_muscles.keys()}
    osim_only_muscles = {i for i in res_osim_muscles.keys() if i not in res_mujoco_muscles.keys()}
    if mujoco_only_muscles:
        print(f"WARNING: Muscles only available in MuJoCo model: {mujoco_only_muscles}")
    if osim_only_muscles:
        print(f"WARNING: Muscles only available in OpenSim model: {osim_only_muscles}")

    res_mujoco_common_muscles = {k: v for k, v in res_mujoco_muscles.items() if k not in mujoco_only_muscles}
    res_osim_common_muscles = {k: v for k, v in res_osim_muscles.items() if k not in osim_only_muscles}
    if not all([np.isclose(v1, v2).all() for (k1, v1), (k2, v2) in zip(res_mujoco_common_muscles.items(), res_osim_common_muscles.items())]):
        input((res_mujoco_muscles, res_osim_muscles))
        print("WARNING: Muscle states do not match between MuJoCo and OpenSim!")

    ## Body state (pt.1: position)
    ### INFO: Run forward kinematics to compute (global) body position corresponding to current qpos:
    mujoco_py.cymj._mj_kinematics(mujoco_env.sim.model, mujoco_env.sim.data)
    res_mujoco_bodies = {mujoco_env.sim.model.body_id2name(i): mujoco_env.sim.data.body_xpos[i] for i in range(mujoco_env.sim.model.nbody)}
    #res_osim_bodies = osim_env.get_state_desc()["body_pos"]  #relative to ground frame
    res_osim_bodies = {body_frame.getName(): [body_frame.getTransformInGround(osim_env.state).p()[i] for i in range(3)] for body_id in range(osim_env.bodySet.getSize()) if (body_frame := osim.Frame.safeDownCast(osim_env.bodySet.get(body_id))) is not None}  #relative to parent frame
    #input((res_mujoco_bodies, res_osim_bodies))

    mujoco_only_bodies = {i for i in res_mujoco_bodies.keys() if i not in res_osim_bodies.keys()}
    osim_only_bodies = {i for i in res_osim_bodies.keys() if i not in res_mujoco_bodies.keys()}
    if mujoco_only_bodies:
        print(f"WARNING: Bodies only available in MuJoCo model: {mujoco_only_bodies}")
    if osim_only_bodies:
        print(f"WARNING: Bodies only available in OpenSim model: {osim_only_bodies}")

    res_mujoco_common_bodies = {k: v for k, v in res_mujoco_bodies.items() if k not in mujoco_only_bodies}
    res_osim_common_bodies = {k: v for k, v in res_osim_bodies.items() if k not in osim_only_bodies}

    body_pos_diffs = [np.linalg.norm(v_m - v_o) for (_, v_m), (_, v_o) in zip(res_mujoco_common_bodies.items(), res_osim_common_bodies.items())]
    if not all([i < 1e-6 for i in body_pos_diffs]):
        input((body_pos_diffs, res_mujoco_common_bodies, res_osim_common_bodies))
        print("WARNING: Body positions do not match between MuJoCo and OpenSim!")

    ## Body state (pt.2: (diagonal) inertia, mass, center of mass)
    # for i in range(mujoco_env.sim.model.nbody):
    #     print((mujoco_env.sim.model.body_id2name(i), mujoco_env.sim.data.cinert[i], mujoco_env.sim.model.body_inertia[i], mujoco_env.sim.model.body_mass[i], mujoco_env.sim.model.body_ipos[i]))
    # for body_id in range(osim_env.bodySet.getSize()):
    #     print((osim_env.bodySet.get(body_id).getName(), [osim_env.bodySet.get(body_id).getInertia().getMoments()[i] for i in range(3)] + [osim_env.bodySet.get(body_id).getInertia().getProducts()[i] for i in range(3)]))
    ### INFO: Run forward simulation to compute center-of-mass inertias for all bodies, corresponding to current qpos:
    mujoco_py.cymj._mj_forward(mujoco_env.sim.model, mujoco_env.sim.data)
    ### INFO: Scale OpenSim/MuJoCo inertias too allow for higher error threshold below, compared to other quantities
    inertia_scale = 1  # 0.001
    res_mujoco_inertias = {mujoco_env.sim.model.body_id2name(i): np.concatenate((inertia_scale *
                                                                               mujoco_env.sim.model.body_inertia[i],
                                                                               [mujoco_env.sim.model.body_mass[i]],
                                                                               mujoco_env.sim.model.body_ipos[i])) for i
                         in range(mujoco_env.sim.model.nbody)}
    ### INFO: OpenSim inertia diagonal entries are sorted in reverse order, as this is internally done in MuJoCo (does this make sense??)
    res_osim_inertias = {
        body_frame.getName(): sorted([inertia_scale * osim_env.bodySet.get(body_id).getInertia().getMoments()[i] for i in range(3)],
            reverse=True) + [osim_env.bodySet.get(body_id).getMass()] + [
                                  osim_env.bodySet.get(body_id).getMassCenter()[i] for i in range(3)] for body_id in
        range(osim_env.bodySet.getSize()) if
        (body_frame := osim.Frame.safeDownCast(osim_env.bodySet.get(body_id))) is not None}
    # input((res_mujoco_inertias, res_osim_inertias))

    mujoco_only_inertias = {i for i in res_mujoco_inertias.keys() if i not in res_osim_inertias.keys()}
    osim_only_inertias = {i for i in res_osim_inertias.keys() if i not in res_mujoco_inertias.keys()}
    # if mujoco_only_inertias:
    #     print(f"WARNING: Bodies only available in MuJoCo model: {mujoco_only_inertias}")
    # if osim_only_inertias:
    #     print(f"WARNING: Bodies only available in OpenSim model: {osim_only_inertias}")

    res_mujoco_common_inertias = {k: v for k, v in res_mujoco_inertias.items() if k not in mujoco_only_inertias}
    res_osim_common_inertias = {k: v for k, v in res_osim_inertias.items() if k not in osim_only_inertias}

    body_inertias_diffs = [np.linalg.norm(v_m - v_o) for (_, v_m), (_, v_o) in
                      zip(res_mujoco_common_inertias.items(), res_osim_common_inertias.items())]
    if not all([i < 2e-3 for i in body_inertias_diffs]):
        input((body_inertias_diffs, res_mujoco_common_inertias, res_osim_common_inertias))
        print("WARNING: Body inertias do not match between MuJoCo and OpenSim!")

    return (res_mujoco_common == res_osim_common) and \
           all([np.isclose(v1, v2).all() for (k1, v1), (k2, v2) in zip(res_mujoco_common_muscles.items(), res_osim_common_muscles.items())]) and \
           all([i < 1e-6 for i in body_pos_diffs]) and \
           all([i < 2e-3 for i in body_inertias_diffs])


def compare_MuJoCo_OpenSim_models(env, osim_filepath):
    """
    Analyzes whether MuJoCo and OpenSim are aligned, such that the same controls/activations result in the same trajectories.
    :param env: Gym environment of MuJoCo model
    :param osim_filepath: OpenSim model filepath
    :return:
    """

    osim_env = OsimModelEnv(osim_filepath, stepsize=0.002)

    # Verify that both models have same number of (muscle) actuators
    mujoco_actuator_indexlist = [i for i in range(env.sim.model.nu) if env.sim.model.actuator_trntype[i] == 3]
    if len(mujoco_actuator_indexlist) != osim_env.get_action_space_size():
        print(f"WARNING: Different number of muscle actuators observed (MuJoCo: {len(mujoco_actuator_indexlist)}, OpenSim: {osim_env.get_action_space_size()}!")

    mujoco_observation = env.reset()
    osim_observation = osim_env.reset()

    #assert env.sim.model.nq == osim_env.model.getNumCoordinates()

    # Set initial OpenSim state
    #init_osim_state = pd.read_csv(os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/joint_angles_flo.mot'), skiprows=10, delimiter="\t", index_col="time", nrows=1).iloc[0].to_dict()
    init_osim_state = pd.read_csv(os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/CMC_Reach8_states_abbrev.sto'), skiprows=6, delimiter="\t", index_col="time", nrows=1).iloc[0].to_dict()
    # init_osim_state["/jointset/elbow/elbow_flexion/value"] = 0.4  #TODO: delete this
    # init_osim_state["/jointset/shoulder0/elv_angle/value"] = 1.5707963267949001  #TODO: delete this
    # init_osim_state["/jointset/shoulder1/shoulder_elv/value"] = 0.4  #TODO: delete this
    # osim_get_coordinate(osim_env, "shoulder_elv").set_locked(True)  #TODO: delete this
    # osim_get_coordinate(osim_env, "elv_angle").set_locked(True)  #TODO: delete this
    #system_state = osim_env.get_state()
    #input(system_state.getY())
    #osim_env.set_state(system_state)
    coordinate_joint_dict = {joint.get_coordinates(j).getSpeedName().split("/")[0]: joint.getName() for i in range(osim_env.jointSet.getSize()) if (joint := osim_env.jointSet.get(i)) is not None for j in range(joint.numCoordinates())}
    coordinate_value_names_list = ["/" + "/".join(("jointset", joint.getName(), joint.get_coordinates(j).getSpeedName().split("/")[0], value_name)) for i in range(osim_env.jointSet.getSize()) if (joint := osim_env.jointSet.get(i)) is not None for j in range(joint.numCoordinates()) for value_name in ("value", "speed")]
    for coordinate_name, coordinate_value in init_osim_state.items():
        if coordinate_name in coordinate_joint_dict:  #e.g., coordinate_name == "sternoclavicular_r2"
            # input((osim_env.jointSet.getStateVariableValue(osim_env.state, "/jointset/sternoclavicular/sternoclavicular_r2/value")))
            osim_env.jointSet.setStateVariableValue(osim_env.state, f"/jointset/{coordinate_joint_dict[coordinate_name]}/{coordinate_name}/value", coordinate_value)
            osim_env.jointSet.setStateVariableValue(osim_env.state, f"/jointset/{coordinate_joint_dict[coordinate_name]}/{coordinate_name}/speed", 0)
            # input(osim_env.get_joint_values())
        elif coordinate_name in coordinate_value_names_list:  #e.g., coordinate_name == "/jointset/sternoclavicular/sternoclavicular_r2/speed"
           osim_env.jointSet.setStateVariableValue(osim_env.state, coordinate_name, coordinate_value)
        else:
            try:  #e.g., coordinate_name == "/forceset/DELT1/activation"
                osim_env.jointSet.setStateVariableValue(osim_env.state, coordinate_name, coordinate_value)
            except RuntimeError:
               print(f"Cannot set initial value for coordinate {coordinate_name}, as it is not found in used OpenSim model.")

    #osim_env.jointSet.setStateVariableValue(osim_env.state, f"/jointset/sternoclavicular/sternoclavicular_r2/value", 0.5)

    osim_env.set_state(osim_env.state)  #IMPORTANT INFO: this is required to send current state stored in osim_env.state to OpenSim Manager! (actually, "manager.initialize(self.state)" would suffice...)
    osim_env.model.realizeReport(osim_env.state)  #IMPORTANT INFO: this is required to make (initial) muscle properties accesible

    # Set MuJoCo joint values to OpenSim joint values
    ## Compute MuJoCo/OpenSim joint values
    res_mujoco = env.get_joint_values()
    res_osim = osim_env.get_joint_values()
    env.sim.data.qvel[:] = np.zeros(env.sim.model.nv)
    env.sim.data.qacc[:] = np.zeros(env.sim.model.nv)
    env.sim.data.qacc_warmstart[:] = np.zeros(env.sim.model.nv)
    for joint_name, value in res_mujoco.items():
        if joint_name in res_osim.keys():
            env.sim.data.qpos[env.sim.model.joint_name2id(joint_name)] = res_osim[joint_name][0]
            env.sim.data.qvel[env.sim.model.joint_name2id(joint_name)] = res_osim[joint_name][1]
            #env.sim.data.qacc[env.sim.model.joint_name2id(joint_name)] = res_osim[joint_name][2]  #setting acceleration should not have any effect in MuJoCo
        else:
            print(f"WARNING: Cannot set values of MuJoCo joint '{joint_name}', as it does not exist in used OpenSim model!")
    res_mujoco = env.get_joint_values()

    # Set MuJoCo muscle values to OpenSim muscle values
    ## Compute MuJoCo/OpenSim muscle values
    res_mujoco = env.get_muscle_values()
    res_osim = osim_env.get_muscle_values()
    env.sim.data.act[:] = np.zeros(env.sim.model.na)
    env.sim.data.act_dot[:] = np.zeros(env.sim.model.na)
    for muscle_name, value in res_mujoco.items():
        if muscle_name in res_osim.keys():
            actuator_id = mujoco_actuator_indexlist.index(env.sim.model.actuator_name2id(muscle_name))
            env.sim.data.act[actuator_id] = res_osim[muscle_name][0]
            env.sim.data.act_dot[actuator_id] = res_osim[muscle_name][1]
            ## -> Set env.sim.data.actuator_length s.t. (env.sim.data.actuator_length[<muscle-id>] - LT) == [OpenSim]"/forceset/<muscle-name>/fiber_length" holds
            ## (for details, see https://mujoco.readthedocs.io/en/latest/modeling.html#muscle-actuators):
            LO = (env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_lengthrange[actuator_id][1]) / (
                         env.sim.model.actuator_gainprm[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][1])
            LT = env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][0] * LO
            env.sim.data.actuator_length[actuator_id] = res_osim[muscle_name][2] + LT
        else:
            print(f"WARNING: Cannot set values of MuJoCo muscle '{muscle_name}', as it does not exist in used OpenSim model!")
    res_mujoco = env.get_muscle_values()

    assert check_model_state_equality(env, osim_env), "ERROR: MuJoCo and OpenSim are not in the same initial state!"  #TODO: uncomment this!

    ##########################################################
    ################### DETERMINE CONTROLS ###################
    ##########################################################
    # VARIANT 1: Use controls computed by StaticOptimization
    # static_optimization(ik_filename=os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/joint_angles_flo.mot'),
    #                     results_dirname=os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/'),
    #                     weight_activations=False)
    #ctrl_samples = pd.read_csv(os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/joint_angles_flo_StaticOptimization_activation.sto'), skiprows=8, delimiter="\t", index_col="time")
    ctrl_samples = pd.read_csv(os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/CMC_Reach8_controls.sto'), skiprows=6, delimiter="\t", index_col="time")

    n_steps = ctrl_samples.shape[0]
    ## Exclude introduced reserve actuators from evaluation
    ctrl_samples = ctrl_samples.loc[:, [('reserve' not in column) & (column not in ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']) for column in ctrl_samples.columns]]
    ### Exclude actuators that are not available in MuJoCo
    mujoco_actuator_names = [env.sim.model.actuator_id2name(id) for id in range(env.sim.model.nu)]
    for actuator_name in ctrl_samples.columns:
        if actuator_name not in mujoco_actuator_names:
            print(f"Cannot set values for actuator {actuator_name}, as it is not found in used MuJoCo model.")
            ctrl_samples = ctrl_samples.drop(actuator_name, axis=1)
    ### Compute actuator indices that match order of OpenSim Data
    mujoco_actuator_indexlist_fromfile = [env.sim.model.actuator_name2id(actuator_name) for actuator_name in ctrl_samples.columns]
    ctrl_samples = ctrl_samples.to_numpy()

    # VARIANT 2: Use random controls
    #n_steps = 3
    #ctrl_samples = [osim_env.action_space.sample() for _ in range(n_steps)]
    ##########################################################

    # Run OpenSim Forward Simulation
    observations_osim = [osim_env.get_joint_values()]
    n_steps = 500
    for i in range(n_steps):
        #assert i == osim_env.istep
        print('OpenSim Forward Simulation - Step #{}/{} ({}s) [{:.2f}%]'.format(i, n_steps, osim_env.state.getTime(), float(i * 100 / n_steps) if n_steps != 0 else 100))
        observation, reward, done, info = osim_env.step(ctrl_samples[i])
        #input((observation, reward, done, info))
        #print(f"STEP: {osim_env.istep}, TIME: {osim_env.state.getTime()}")
        observations_osim.append(osim_env.get_joint_values())
        #input((observations_osim[-1]))

    # Run MuJoCo Forward Simulation
    observations_mujoco = [env.get_joint_values()]
    #input(env.sim.data.act[:])
    #input(env.sim.data.qacc[:])
    #input(env.sim.data.qvel[:])
    #input([(contact.geom1, contact.geom2, contact.dist) for contact in env.sim.data.contact])
    for i in range(n_steps):
        print('MuJoCo Forward Simulation - Step #{}/{} [{:.2f}%]'.format(i, n_steps, float(i * 100 / n_steps) if n_steps != 0 else 100))
        env.sim.data.ctrl[:] = np.zeros(env.sim.model.nu)
        env.sim.data.ctrl[mujoco_actuator_indexlist_fromfile] = ctrl_samples[i]
        #input(env.sim.data.ctrl[:])
        #mujoco_py.cymj._mj_fwdActuation(env.sim.model, env.sim.data)
        #mujoco_py.cymj._mj_fwdVelocity(env.sim.model, env.sim.data)
        env.sim.step()

        # # adjust virtual joints according to active constraints:
        # for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
        #         env.sim.model.eq_obj1id[(env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1)],
        #         env.sim.model.eq_obj2id[(env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1)],
        #         env.sim.model.eq_data[(env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1), 4::-1]):
        #     env.sim.data.qpos[virtual_joint_id] = np.polyval(poly_coefs, env.sim.data.qpos[physical_joint_id])

        # res_mujoco_bodies = {env.sim.model.body_id2name(i): env.sim.data.body_xpos[i] for i in
        #                      range(env.sim.model.nbody)}
        # # res_osim_bodies = osim_env.get_state_desc()["body_pos"]  #relative to ground frame
        # res_osim_bodies = {body.getName(): [body.getTransformInGround(osim_env.state).p()[i] for i in range(3)] for i in
        #                    range(osim_env.bodySet.getSize()) if (body := osim.Frame.safeDownCast(
        #         osim_env.bodySet.get(i))) is not None}  # relative to parent frame
        # input((res_mujoco_bodies, res_osim_bodies))

        #env.render()
        #input(env.sim.data.act[:])
        #input(env.sim.data.qacc[:])
        #print(env.sim.data.qvel[env.sim.model.joint_name2id("elbow_flexion")])
        #input((env.sim.data.qacc[env.sim.model.joint_name2id("elbow_flexion")], env.sim.data.qacc_unc[env.sim.model.joint_name2id("elbow_flexion")]))
        #input(env.sim.data.qvel[:])
        #input((env.sim.data.qpos, env.sim.data.qvel, env.sim.data.act, env.sim.data.ctrl))
        observations_mujoco.append(env.get_joint_values())

    input(([i["shoulder_elv"][0] for i in observations_mujoco][::n_steps//10]))
    input(([i["shoulder_elv"][0] for i in observations_osim][::n_steps//10]))
    input((observations_mujoco[-1], observations_osim[-1]))
