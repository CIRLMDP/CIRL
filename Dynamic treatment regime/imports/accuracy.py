#######################################################################################################################
#   file name: accuracy
#
#   description:
#   this file defines 2 functions that are used to calculate accuracy measures.
#######################################################################################################################
# imports:
#######################################################################################################################
import numpy as np
#######################################################################################################################
# calculates accuracy given expert and agent policies:
def accuracy_mesure(agent_poicy, expert_policy):
    actions_diff = agent_poicy - expert_policy
    diffs = np.count_nonzero(actions_diff)
    same = actions_diff.size - diffs
    return float(same)/float(actions_diff.size)

#######################################################################################################################
# calculates "distance" between actions - assuming binned actions of 5X5 given expert and agent policies:
def actions_distance(agent_poicy, expert_policy):
    diff1 = abs(np.remainder(agent_poicy,5) - np.remainder(expert_policy,5))
    diff2 = abs(np.floor_divide(agent_poicy,5) - np.floor_divide(expert_policy,5))
    return np.sum(diff1 + diff2)
#######################################################################################################################
