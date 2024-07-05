whac-a-mole:

  v1.1.0
    - add bonus term for target contacts, i.e., unsuccessful target hits (usually because the velocity constraint was not met)

  v1.1.0l [non-permanent changes!]
    - variant of v1.1.0 which only spawns targets in the first column of the target grid (i.e., only on the left)

  v1.1.1 [non-permanent changes!]
    - use splines to define distance reward to currently shown targets and ensure that distance reward terms are non-negative
    - NOTE: in subsequent versions, distance reward terms were reset to the original exponential definition

  v1.1.2 [DEFAULT BEHAVIOR: was used later on also in updated versions of v1.1.0 etc. ...]
    - variant of v1.1.0, where the target contact bonus term is scaled by the hitting velocity (which is defined as z-axis velocity of hammer)

  v1.1.3
    - new game option: allow to automatically increase probability of targets with low hitting rate in previous run/episode

  v1.1.4
    - as v1.1.0, but with options to v1.1.3 and sparse rewards (i.e., purely extrinsic game rewards) (-> use "-adaptive" and "-sparse" flags in built apps, respectively)

  v1.1.5
    - bugfixes in contact bonus reward term introduced in v1.1.2 (use correct hitting velocity for low conditions; provide contact bonus only once per target spawned)
    - allow to send arbitrary Unity variables accessible from RLEnv to Python and log them their using WandB
