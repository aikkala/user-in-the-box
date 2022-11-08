import os
from uitb_evaluate.trajectory_data import TrajectoryData_RL, TrajectoryData_STUDY
from uitb_evaluate.evaluate_main import trajectoryplot
from uitb_evaluate.evaluate_summarystatistics import sumstatsplot
from utils import check_study_dataset_dir

DIRNAME_SIMULATION = os.path.abspath("../")
DIRNAME_STUDY = os.path.abspath("../study/")
PLOTS_DIR = os.path.abspath("_generated_plots/")  #overwrites PLOTS_DIR_DEFAULT from uitb_evaluate.trajectory_data

if __name__ == "__main__":
    check_study_dataset_dir(DIRNAME_STUDY)
    #################################

    _active_parts = [1, 2, 3, 4, 5]

    ########################################

    if 1 in _active_parts:
        # Figure 5a/Suppl. Figure 2/Suppl. Figure 3
        _subtitle = """PART 1: RL - ISO POINTING (Summary statistics)"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        filename = "iso-pointing-U1-patch-v1-dwell-random/evaluate/random/"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}/state_log.pickle")

        # Preselect some episodes:
        EPISODE_IDS = None  # "2"#.zfill(len(list(data.keys())[0].split("episode_")[-1]))  #.zfill(3 + (("100episodes" not in filepath) and ("state_log" not in filepath) and ("TwoLevel" not in filepath)) - ("TwoLevel" in filepath))
        ## only if REPEATED_MOVEMENTS:
        MOVEMENT_IDS = None  # .zfill(len(list(data.keys())[0].split("movement_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True
        RADIUS_IDS = None  # "1".zfill(len(list(data.keys())[0].split("radius_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True

        USER_ID = "U1"  # (only used if SHOW_STUDY == True)
        # AGGREGATE_TRIALS = True  ##(only used if SHOW_STUDY == True)

        #########################

        REPEATED_MOVEMENTS = "repeated-movements" in filename

        # Preprocess simulation trajectories (RL environment):
        trajectories_SIMULATION = TrajectoryData_RL(DIRNAME_SIMULATION, filename, REPEATED_MOVEMENTS=REPEATED_MOVEMENTS)
        trajectories_SIMULATION.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS,
                                           split_trials="tracking" not in filename and "driving" not in filename)

        #########################

        PLOTTING_ENV = "RL-UIB"

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        REMOVE_FAILED = True
        REMOVE_OUTLIERS = False

        EFFECTIVE_PROJECTION_PATH = True
        USE_TARGETBOUND_AS_DIST = False
        DWELL_TIME = 0.5 if "iso-" in filename else 0.3  # only used if USE_TARGETBOUND_AS_DIST == False
        MAX_TIME = 4.0  # only used if REMOVE_FAILED == True

        #PLOT_TYPE = "alldata"  # "alldata", "boxplot", "mean_groups", "meandata", "density_ID"
        BOXPLOT_category = "ID"
        BOXPLOT_nbins = 5
        BOXPLOT_qbins = True  # whether to use quantile-based bins (i.e., same number of samples per bin) or range-based bins (i.e., same length of each bin interval)

        DENSITY_group_nIDbins = 1  # number of ID groups
        DENSITY_group_IDbin_ID = 0  # index of ID group (between 0 and DENSITY_group_nIDbins-1) for which a density plot of movement times is created (only used if PLOT_TYPE == "density_ID")
        DENSITY_group_nMTbins = 50

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        # Font sizes etc.
        plot_width = "sumstats"

        ####

        for PLOT_TYPE in ("meandata", "density_ID", "alldata"):
            sumstatsplot(filename, trajectories_SIMULATION,
                         REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                         MOVEMENT_IDS=MOVEMENT_IDS,
                         RADIUS_IDS=RADIUS_IDS,
                         EPISODE_IDS=EPISODE_IDS,
                         EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                         USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                         DWELL_TIME=DWELL_TIME,
                         MAX_TIME=MAX_TIME,
                         PLOT_TYPE=PLOT_TYPE,
                         BOXPLOT_category=BOXPLOT_category,
                         BOXPLOT_nbins=BOXPLOT_nbins,
                         BOXPLOT_qbins=BOXPLOT_qbins,
                         DENSITY_group_nIDbins=DENSITY_group_nIDbins,
                         DENSITY_group_IDbin_ID=DENSITY_group_IDbin_ID,
                         DENSITY_group_nMTbins=DENSITY_group_nMTbins,
                         plot_width=plot_width,
                         STORE_PLOT=STORE_PLOT, PLOTS_DIR=PLOTS_DIR,
                         STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 2 in _active_parts:
        # Figure 5b
        _subtitle = """PART 2: RL - EFFECT OF TARGET SIZE ON END-EFFECTOR MOVEMENT"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        filename = "iso-pointing-U1-patch-v1-dwell-random/evaluate/repeated-movements/"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}/state_log.pickle")

        # Preselect some episodes:
        EPISODE_IDS = None  # "2"#.zfill(len(list(data.keys())[0].split("episode_")[-1]))  #.zfill(3 + (("100episodes" not in filepath) and ("state_log" not in filepath) and ("TwoLevel" not in filepath)) - ("TwoLevel" in filepath))
        ## only if REPEATED_MOVEMENTS:
        MOVEMENT_IDS = None  # .zfill(len(list(data.keys())[0].split("movement_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True
        RADIUS_IDS = None  # "1".zfill(len(list(data.keys())[0].split("radius_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True

        USER_ID = "U1"  # (only used if SHOW_STUDY == True)
        # AGGREGATE_TRIALS = True  ##(only used if SHOW_STUDY == True)

        #########################

        REPEATED_MOVEMENTS = "repeated-movements" in filename

        # Preprocess simulation trajectories (RL environment):
        trajectories_SIMULATION = TrajectoryData_RL(DIRNAME_SIMULATION, filename, REPEATED_MOVEMENTS=REPEATED_MOVEMENTS)
        trajectories_SIMULATION.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS,
                                           EPISODE_IDS=EPISODE_IDS,
                                           split_trials="tracking" not in filename and "driving" not in filename)

        # Preprocess simulation trajectories (ISO Task User Study):
        trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID)
        trajectories_STUDY.preprocess()

        #########################

        PLOTTING_ENV = "RL-UIB"

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = range(1, 9)  # [i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = None  # [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # range(1,8)  #corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = None  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = ["episode",
                            "movement"]  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 3  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        # DWELL_TIME = 0.3  #tail of the trajectories that is not shown (in seconds)
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = False
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, None,
                       None, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT, PLOTS_DIR=PLOTS_DIR,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 3 in _active_parts:
        # Figure 6
        _subtitle = """PART 3: RL - ISO POINTING TASK (Simulation vs. MinJerk vs. Human Data)"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        filename = "iso-pointing-U1-patch-v1-dwell-random/evaluate/ISO/"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}/state_log.pickle")

        # Preselect some episodes:
        EPISODE_IDS = None  # "2"#.zfill(len(list(data.keys())[0].split("episode_")[-1]))  #.zfill(3 + (("100episodes" not in filepath) and ("state_log" not in filepath) and ("TwoLevel" not in filepath)) - ("TwoLevel" in filepath))
        ## only if REPEATED_MOVEMENTS:
        MOVEMENT_IDS = None  # .zfill(len(list(data.keys())[0].split("movement_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True
        RADIUS_IDS = None  # "1".zfill(len(list(data.keys())[0].split("radius_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True

        USER_ID = "U1"  # (only used if SHOW_STUDY == True)
        # AGGREGATE_TRIALS = True  ##(only used if SHOW_STUDY == True)

        #########################

        REPEATED_MOVEMENTS = "repeated-movements" in filename

        # Preprocess simulation trajectories (RL environment):
        trajectories_SIMULATION = TrajectoryData_RL(DIRNAME_SIMULATION, filename, REPEATED_MOVEMENTS=REPEATED_MOVEMENTS)
        trajectories_SIMULATION.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS,
                                           EPISODE_IDS=EPISODE_IDS,
                                           split_trials="tracking" not in filename and "driving" not in filename)

        # Preprocess simulation trajectories (ISO Task User Study):
        trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID)
        trajectories_STUDY.preprocess()

        #########################

        PLOTTING_ENV = "RL-UIB"

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # [i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = [7]

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = True  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = range(1, 4)  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = range(1,
                          14)  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = []  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = False  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 3  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        # DWELL_TIME = 0.3  #tail of the trajectories that is not shown (in seconds)
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = False
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = True
        SHOW_STUDY = False
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, None,
                       None, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       STORE_PLOT=STORE_PLOT, PLOTS_DIR=PLOTS_DIR,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

        PLOT_ENDEFFECTOR = False
        SHOW_MINJERK = False
        SHOW_STUDY = True
        for JOINT_ID in range(5):
            trajectoryplot(PLOTTING_ENV, USER_ID, None,
                           None, filename, trajectories_SIMULATION,
                           trajectories_STUDY=trajectories_STUDY,
                           REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                           MOVEMENT_IDS=MOVEMENT_IDS,
                           RADIUS_IDS=RADIUS_IDS,
                           EPISODE_IDS=EPISODE_IDS,
                           r1_FIXED=r1_FIXED,
                           r2_FIXED=r2_FIXED,
                           EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                           USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                           MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                           TARGET_IDS=TARGET_IDS,
                           TRIAL_IDS=TRIAL_IDS,
                           META_IDS=META_IDS,
                           N_MOVS=N_MOVS,
                           AGGREGATION_VARS=AGGREGATION_VARS,
                           PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                           PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                           JOINT_ID=JOINT_ID,
                           PLOT_DEVIATION=PLOT_DEVIATION,
                           NORMALIZE_TIME=NORMALIZE_TIME,
                           # DWELL_TIME=#DWELL_TIME,
                           PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                           PLOT_VEL_ACC=PLOT_VEL_ACC,
                           PLOT_RANGES=PLOT_RANGES,
                           CONF_LEVEL=CONF_LEVEL,
                           SHOW_MINJERK=SHOW_MINJERK,
                           SHOW_STUDY=SHOW_STUDY,
                           STUDY_ONLY=STUDY_ONLY,
                           ENABLE_LEGENDS_AND_COLORBARS=JOINT_ID == 2,
                           ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                           STORE_PLOT=STORE_PLOT, PLOTS_DIR=PLOTS_DIR,
                           STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 4 in _active_parts:
        # Figure 7
        _subtitle = """PART 4: RL - TRACKING TASK (Comparing multiple frequencies)"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        filename = "tracking-v1-patch-v1/evaluate/max-freq-"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}/state_log.pickle")

        # Preselect some episodes:
        EPISODE_IDS = None  # "2"#.zfill(len(list(data.keys())[0].split("episode_")[-1]))  #.zfill(3 + (("100episodes" not in filepath) and ("state_log" not in filepath) and ("TwoLevel" not in filepath)) - ("TwoLevel" in filepath))
        ## only if REPEATED_MOVEMENTS:
        MOVEMENT_IDS = None  # .zfill(len(list(data.keys())[0].split("movement_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True
        RADIUS_IDS = None  # "1".zfill(len(list(data.keys())[0].split("radius_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True

        USER_ID = "U1"  # (only used if SHOW_STUDY == True)
        # AGGREGATE_TRIALS = True  ##(only used if SHOW_STUDY == True)

        #########################

        REPEATED_MOVEMENTS = "repeated-movements" in filename

        # Preprocess simulation trajectories (RL environment):
        trajectories_SIMULATION = []
        for i in ["005", "025", "050", "100"]:
            trajectories_SIMULATION.append(
                TrajectoryData_RL(DIRNAME_SIMULATION, f"{filename}{i}", REPEATED_MOVEMENTS=REPEATED_MOVEMENTS))
            trajectories_SIMULATION[-1].preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS,
                                                   EPISODE_IDS=EPISODE_IDS,
                                                   split_trials="tracking" not in filename and "driving" not in filename)
            trajectories_SIMULATION[-1].max_frequency = int(i) / 100

        # Preprocess simulation trajectories (ISO Task User Study):
        trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID)
        trajectories_STUDY.preprocess()

        #########################

        PLOTTING_ENV = "RL-UIB"

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # [i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = None

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = None  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = [
            "episode"]  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = True  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 3  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        # DWELL_TIME = 0.3  #tail of the trajectories that is not shown (in seconds)
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = True
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = False
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        # Font sizes etc.
        plot_width = "halfpage"

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, None,
                       None, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       plot_width=plot_width,
                       STORE_PLOT=STORE_PLOT, PLOTS_DIR=PLOTS_DIR,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################

    if 5 in _active_parts:
        # Figure 8
        _subtitle = """PART 5: RL - REMOTE TASK (Distance to Joystick/Target)"""
        print(f"\n\n+++++++++++++++++++++++++++++++++++++\n{_subtitle}\n++++++++++++++++++++++++++++++++++++++\n")

        filename = "driving-model-v2-patch-v1-newest-location-no-termination/evaluate"
        filepath = os.path.join(DIRNAME_SIMULATION, f"{filename}/state_log.pickle")

        # Preselect some episodes:
        EPISODE_IDS = None  # "2"#.zfill(len(list(data.keys())[0].split("episode_")[-1]))  #.zfill(3 + (("100episodes" not in filepath) and ("state_log" not in filepath) and ("TwoLevel" not in filepath)) - ("TwoLevel" in filepath))
        ## only if REPEATED_MOVEMENTS:
        MOVEMENT_IDS = None  # .zfill(len(list(data.keys())[0].split("movement_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True
        RADIUS_IDS = None  # "1".zfill(len(list(data.keys())[0].split("radius_")[-1].split("__")[0]))  # only used if REPEATED_MOVEMENTS == True

        USER_ID = "U1"  # (only used if SHOW_STUDY == True)
        # AGGREGATE_TRIALS = True  ##(only used if SHOW_STUDY == True)

        #########################

        REPEATED_MOVEMENTS = "repeated-movements" in filename

        # Preprocess simulation trajectories (RL environment):
        trajectories_SIMULATION = TrajectoryData_RL(DIRNAME_SIMULATION, filename, REPEATED_MOVEMENTS=REPEATED_MOVEMENTS)
        trajectories_SIMULATION.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS, EPISODE_IDS=EPISODE_IDS,
                                           split_trials=False, endeffector_name="car")
        trajectories_SUPPLEMENTARY = TrajectoryData_RL(DIRNAME_SIMULATION, filename,
                                                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS)
        trajectories_SUPPLEMENTARY.preprocess(MOVEMENT_IDS=MOVEMENT_IDS, RADIUS_IDS=RADIUS_IDS,
                                              EPISODE_IDS=EPISODE_IDS, split_trials=False,
                                              endeffector_name="fingertip", target_name="joystick")

        # Preprocess simulation trajectories (ISO Task User Study):
        trajectories_STUDY = TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=USER_ID)
        trajectories_STUDY.preprocess()

        # # Preprocess simulation trajectories (ISO Task User Study - Other Users (for between-user comparison)):
        # trajectories_STUDY_OTHERS = []
        # for uid in [f"U{i}" for i in range(1, 7)]:
        #     if uid != USER_ID:
        #         trajectories_STUDY_OTHERS.append(
        #             TrajectoryData_STUDY(DIRNAME_STUDY, USER_ID=uid))
        #         trajectories_STUDY_OTHERS[-1].preprocess()

        #########################

        PLOTTING_ENV = "RL-UIB"

        #################################

        #################################
        ### SET PLOTTING PARAM VALUES ###
        #################################

        # WHICH PARTS OF DATASET? #only used if PLOTTING_ENV == "RL-UIB"
        MOVEMENT_IDS = None  # [i for i in range(10) if i != 1]
        RADIUS_IDS = None
        EPISODE_IDS = None

        r1_FIXED = None  # r1list[5]  #only used if PLOTTING_ENV == "MPC-costweights"
        r2_FIXED = None  # r2list[-1]  #only used if PLOTTING_ENV == "MPC-costweights"

        # TASK_CONDITION_LIST_SELECTED = ["Virtual_Cursor_ID_ISO_15_plane", "Virtual_Cursor_Ergonomic_ISO_15_plane"]  #only used if PLOTTING_ENV == "MPC-taskconditions"

        # WHAT TO COMPUTE?
        EFFECTIVE_PROJECTION_PATH = (
                PLOTTING_ENV == "RL-UIB")  # if True, projection path connects effective initial and final position instead of nominal target center positions
        USE_TARGETBOUND_AS_DIST = False  # True/False or "MinJerk-only"; if True, only plot trajectory until targetboundary is reached first (i.e., until dwell time begins); if "MinJerk-only", complete simulation trajectories are shown, but MinJerk trajectories are aligned to simulation trajectories without dwell time
        MINJERK_USER_CONSTRAINTS = True

        # WHICH/HOW MANY MOVS?
        """
        IMPORTANT INFO:
        if isinstance(trajectories, TrajectoryData_RL):
            -> TRIAL_IDS/META_IDS/N_MOVS are (meta-)indices of respective episode (or rather, of respective row of trajectories.indices)
            -> TRIAL_IDS and META_IDS are equivalent
        if isinstance(trajectories, TrajectoryData_STUDY) or isinstance(trajectories, TrajectoryData_MPC):
            -> TRIAL_IDS/META_IDS/N_MOVS are (global) indices of entire dataset
            -> TRIAL_IDS correspond to trial indices assigned during user study (last column of trajectories.indices), while META_IDS correspond to (meta-)indices of trajectories.indices itself (i.e., if some trials were removed during previous pre-processing steps, TRIAL_IDS and META_IDS differ!)
        In general, only the first not-None parameter is used, in following order: TRIAL_IDS, META_IDS, N_MOVS.
        """
        TARGET_IDS = None  # corresponds to target IDs (if PLOTTING_ENV == "RL-UIB": only for iso-task)
        TRIAL_IDS = None  # [i for i in range(65) if i not in range(28, 33)] #range(1,4) #corresponds to trial IDs [last column of self.indices] or relative (meta) index per row in self.indices; either a list of indices, "different_target_sizes" (choose N_MOVS conditions with maximum differences in target size), or None (use META_IDS)
        META_IDS = None  # index positions (i.e., sequential numbering of trials [aggregated trials, if AGGREGATE_TRIALS==True] in indices, without counting removed outliers); if None: use N_MOVS
        N_MOVS = None  # number of movements to visualize (only used, if TRIAL_IDS and META_IDS are both None (or TRIAL_IDS == "different_target_sizes"))
        AGGREGATION_VARS = [
            "episode"]  # ["all"]  #["episode", "movement"]  #["episode", "targetoccurrence"]  #["targetoccurrence"] #["episode", "movement"]  #["episode", "radius", "movement", "target", "targetoccurrence"]

        # WHAT TO PLOT?
        PLOT_TRACKING_DISTANCE = True  # if True, plot distance between End-effector and target position instead of position (only reasonable for tracking tasks)
        PLOT_ENDEFFECTOR = True  # if True plot End-effector position and velocity, else plot qpos and qvel for joint with index JOINT_ID (see independent_joints below)
        JOINT_ID = 3  # only used if PLOT_ENDEFFECTOR == False
        PLOT_DEVIATION = False  # only if PLOT_ENDEFFECTOR == True

        # HOW TO PLOT?
        NORMALIZE_TIME = False
        # DWELL_TIME = 0.3  #tail of the trajectories that is not shown (in seconds)
        PLOT_TIME_SERIES = True  # if True plot Position/Velocity/Acceleration Time Series, else plot Phasespace and Hooke plots
        PLOT_VEL_ACC = False  # if True, plot Velocity and Acceleration Time Series, else plot Position and Velocity Time Series (only used if PLOT_TIME_SERIES == True)
        PLOT_RANGES = True
        CONF_LEVEL = "min/max"  # might be between 0 and 1, or "min/max"; only used if PLOT_RANGES==True

        # WHICH BASELINE?
        SHOW_MINJERK = False
        SHOW_STUDY = False
        STUDY_ONLY = False  # only used if PLOTTING_ENV == "MPC-taskconditions"

        # PLOT (WHICH) LEGENDS AND COLORBARS?
        ENABLE_LEGENDS_AND_COLORBARS = True  # if False, legends (of axis 0) and colobars are removed
        ALLOW_DUPLICATES_BETWEEN_LEGENDS = False  # if False, legend of axis 1 only contains values not included in legend of axis 0

        # STORE PLOT?
        STORE_PLOT = True
        STORE_AXES_SEPARATELY = True  # if True, store left and right axis to separate figures

        # Font sizes etc.
        plot_width = "halfpage"

        ####

        trajectoryplot(PLOTTING_ENV, USER_ID, None,
                       None, filename, trajectories_SIMULATION,
                       trajectories_STUDY=trajectories_STUDY,
                       trajectories_SUPPLEMENTARY=trajectories_SUPPLEMENTARY,
                       REPEATED_MOVEMENTS=REPEATED_MOVEMENTS,
                       MOVEMENT_IDS=MOVEMENT_IDS,
                       RADIUS_IDS=RADIUS_IDS,
                       EPISODE_IDS=EPISODE_IDS,
                       r1_FIXED=r1_FIXED,
                       r2_FIXED=r2_FIXED,
                       EFFECTIVE_PROJECTION_PATH=EFFECTIVE_PROJECTION_PATH,
                       USE_TARGETBOUND_AS_DIST=USE_TARGETBOUND_AS_DIST,
                       MINJERK_USER_CONSTRAINTS=MINJERK_USER_CONSTRAINTS,
                       TARGET_IDS=TARGET_IDS,
                       TRIAL_IDS=TRIAL_IDS,
                       META_IDS=META_IDS,
                       N_MOVS=N_MOVS,
                       AGGREGATION_VARS=AGGREGATION_VARS,
                       PLOT_TRACKING_DISTANCE=PLOT_TRACKING_DISTANCE,
                       PLOT_ENDEFFECTOR=PLOT_ENDEFFECTOR,
                       JOINT_ID=JOINT_ID,
                       PLOT_DEVIATION=PLOT_DEVIATION,
                       NORMALIZE_TIME=NORMALIZE_TIME,
                       # DWELL_TIME=#DWELL_TIME,
                       PLOT_TIME_SERIES=PLOT_TIME_SERIES,
                       PLOT_VEL_ACC=PLOT_VEL_ACC,
                       PLOT_RANGES=PLOT_RANGES,
                       CONF_LEVEL=CONF_LEVEL,
                       SHOW_MINJERK=SHOW_MINJERK,
                       SHOW_STUDY=SHOW_STUDY,
                       STUDY_ONLY=STUDY_ONLY,
                       ENABLE_LEGENDS_AND_COLORBARS=ENABLE_LEGENDS_AND_COLORBARS,
                       ALLOW_DUPLICATES_BETWEEN_LEGENDS=ALLOW_DUPLICATES_BETWEEN_LEGENDS,
                       plot_width=plot_width,
                       STORE_PLOT=STORE_PLOT, PLOTS_DIR=PLOTS_DIR,
                       STORE_AXES_SEPARATELY=STORE_AXES_SEPARATELY)

    ########################################
