v1.1.0
- add bonus term for target contacts, i.e., unsuccessful target hits (usually because the velocity constraint was not met)

v1.1.1 [non-permanent changes!]
- use splines to define distance reward to currently shown targets and ensure that distance reward terms are non-negative
- NOTE: in subsequent versions, distance reward terms were reset to the original exponential definition

v1.1.0l [non-permanent changes!]
- variant of v1.1.0 which only spawns targets in the first column of the target grid (i.e., only on the left)

v1.1.2 [non-permanent changes!]
- variant of v1.1.0, where the target contact bonus term is scaled by the hitting velocity (which is defined as z-axis velocity of hammer)
