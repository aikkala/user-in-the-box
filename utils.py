import xmltodict

import numpy as np

class opensim_file(object):
  def __init__(self, filepath):
    # Read filepath and parse it
    with open(filepath) as f:
      text = f.read()
    self.opensim_xml = xmltodict.parse(text)
    self.osim_version = self.opensim_xml["OpenSimDocument"]["@Version"]
    for setname in ["BodySet", "ConstraintSet", "ForceSet", "MarkerSet", "ContactGeometrySet"]:
      setattr(self, setname, self.opensim_xml["OpenSimDocument"]["Model"][setname])

  def __getattr__(self, name):
    for typeset in ["BodySet", "ConstraintSet", "ForceSet", "MarkerSet", "ContactGeometrySet"]:
        keyqueue_list = self._itemexists(getattr(self, typeset), name)
        if keyqueue_list:
            newattr = self._valuesdict(name, typeset, keyqueue_list)
            setattr(self, name, newattr)  #avoid re-computation of this attribute
            return newattr

  def _itemexists(self, obj, key, keyqueue=""):
    if key in obj:
        return key
    for k, v in obj.items():
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

def strinterval_to_nparray(interval):
  insplit = interval.split(" ")
  return np.array([float(inval) for inval in insplit], dtype=float)