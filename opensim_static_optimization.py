import os.path

import numpy as np
import pandas as pd
import opensim
import scipy.io as sio
import time

import faulthandler; faulthandler.enable()

#input('This was a test to check whether opensim is loaded correctly.')

def static_optimization(ik_filename=os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/joint_angles_flo.mot'),
                        results_dirname=os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/'),
                        weight_activations=True):

    #model_filename = os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/UpperExtremity1Scaled.osim')
    #model_filename = os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/MoBL_ARMS_wrist4.1_v3.osim')
    # model_filename = os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/MoBL_ARMS_wrist4.1_v3_unscaled.osim')
    model_filename = os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/MOBL_ARMS_fixed_41.osim')

    # ik_filename = os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/SelectedTrials/P3/S2T4_Cropped.mot')

    model = opensim.Model(model_filename)
    reserve_actuator_set = opensim.ForceSet(os.path.expanduser('~/user-in-the-box/UIB/envs/mobl_arms/models/ActuatorsUpExDyn.xml'))
    reserve_actuator_set.setMemoryOwner(False)  # model will be the owner
    for reserve_actuator_id in range(reserve_actuator_set.getSize()):
        model.updForceSet().append(reserve_actuator_set.get(reserve_actuator_id))

    # construct static optimization
    motion = opensim.Storage(ik_filename)
    static_optimization = opensim.StaticOptimization()
    static_optimization.setStartTime(motion.getFirstTime())
    static_optimization.setEndTime(motion.getLastTime())
    static_optimization.setUseModelForceSet(True)
    static_optimization.setUseMusclePhysiology(True)
    static_optimization.setActivationExponent(2)
    static_optimization.setConvergenceCriterion(0.0001)
    static_optimization.setMaxIterations(100)
    model.addAnalysis(static_optimization)

    name = '.'.join(ik_filename.split('/')[-1].split('.')[:-1])

    # analysis
    analysis = opensim.AnalyzeTool(model)
    #analysis.setForceSetFiles(opensim.ArrayStr(os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/ActuatorsUpExDyn.xml')))
    analysis.setName(name)
    analysis.setModel(model)
    analysis.setInitialTime(motion.getFirstTime())
    analysis.setFinalTime(0.02)  #motion.getLastTime())
    analysis.setLowpassCutoffFrequency(6)
    analysis.setCoordinatesFileName(ik_filename)
    #analysis.setExternalLoadsFileName(os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/ActuatorsUpExDyn.xml'))
    analysis.setLoadModelAndInput(True)
    analysis.setResultsDir(results_dirname)

    start_time = time.time()
    analysis.run()
    #
    # print(model.getWorkingState().getQ())
    # print(model.getWorkingState().getU())
    # print(model.getWorkingState().getZ())
    # print(model.getWorkingState().getY())
    # print(model.getWorkingState().getNQ())
    # print(model.getWorkingState().getNU())
    # print(model.getWorkingState().getNY())

    #activation_storage = static_optimization.getActivationStorage()
    #activation_storage_table = activation_storage.exportToTable()
    #print(activation_storage.getData(2,test ))
    #input('ijij')

    print('Static optimization done (duration: {:.2f}s).'.format(time.time() - start_time))

    results = pd.read_csv(results_dirname + '{}_StaticOptimization_activation.sto'.format(name), skiprows=8, delimiter="\t", index_col="time")

    print(results)
    print(results.columns)

    # Exclude introduced reserve actuators from evaluation:
    results = results.loc[:, [('reserve' not in column) & (column not in ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']) for column in results.columns]]

    dt = results.index.to_series().diff().mean()

    if weight_activations:
        maximalmuscleforce_raw = sio.loadmat(os.path.expanduser('~/reacher_sg/ReacherSG/_external_data/MuscleProperties.mat'))
        maximalmuscleforce = np.array([maximalmuscleforce_raw['MuscleProperties']['mass'][0][i][0][0] for i in
                                       range(maximalmuscleforce_raw['MuscleProperties']['mass'][0].shape[0])])
        integrated_energy = ((results * maximalmuscleforce).sum(axis=1) * dt).sum()
    else:
        integrated_energy = (np.linalg.norm(results, axis=1) * dt).sum()

    print('Norm of activations integrated over time: {}.'.format(integrated_energy))
    return integrated_energy

if __name__ == '__main__':
    static_optimization()