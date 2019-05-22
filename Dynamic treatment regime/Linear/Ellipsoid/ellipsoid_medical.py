#######################################################################################################################
#   file name: ellipsoid_medical
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data.
#   then runs the ellipsoid method to solve the linear CIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../imports')

import os
import numpy as np
import scipy.io as sio
from ICMDP import *
from sklearn.cluster import KMeans
import random
import pickle
from Ellipsoid import *
from accuracy import *

#######################################################################################################################
# data savings
#######################################################################################################################
valuesindex = 0
if not os.path.exists('obj'):
    os.mkdir('obj')

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#######################################################################################################################
# HYPER PARAMS
#######################################################################################################################
gamma=0.9
nclusters=500        # amount clusters
n_static=8          # 8-amount of static features
nActions = 25
jump = 3

# simulation parameters:
iters = 1000
epsilon = 5e-5
repeats = 10
tol = 1e-3
RUN_TEST = True

#######################################################################################################################
# construct the CMDP
#######################################################################################################################
data = sio.loadmat('../../data/normalized_data.mat')

ntraj = len(data['normalized_data'])
r = []
a = []
phi=[]
m=[]
state_features = []
traj_len = []

# load real W:
real_W = np.load("../../data/realW.npy")
real_W /= np.linalg.norm(np.reshape(real_W,real_W.size),np.inf)

# get data
expert_contexts =[]
for i in range(ntraj):
    r.append(data['normalized_data'][i][0][0][0][2][0]) #r[i] (1, )
    m.append(data['normalized_data'][i][0][0][0][6][0]) #r[i] (1, )
    a.append(data['normalized_data'][i][0][0][0][4]) #a[i] (74,1)
    phi.append(data['normalized_data'][i][0][0][0][3]) #phi[i] (74,49)
    traj_len.append(phi[i].shape[0])
    non_norm_context = phi[i][0,0:n_static] + 1
    expert_contexts.append( non_norm_context/np.sum(non_norm_context) )

for i in range(ntraj):
    j = 0
    while j < traj_len[i]:
        state_features.append(phi[i][j,n_static:])
        j += jump

# uncomment to re-cluster:
# kmeans_states = KMeans(n_clusters=nclusters, random_state=0).fit(state_features)
# Kmclusters = kmeans_states.cluster_centers_
# Kmlabels = kmeans_states.labels_

# comment to re-cluster:
Kmclusters = np.load("../../data/Kmeans_clusters.npy")
Kmlabels = np.load("../../data/Kmeans_labels.npy")

dim_features = len(phi[0][0,n_static:]) + 1
F = Features(dim_features=dim_features )
feature = np.zeros(dim_features)
for i in range(nclusters):
    feature[:-1] = Kmclusters[i,:]
    F.add_feature(feature=feature)

# add features of 2 possible final states (and another one for un-taken actions):
feature = np.zeros(dim_features)
feature[-1] = -0.5
F.add_feature(feature=feature)
feature[-1] = 0.5
F.add_feature(feature=feature)
F.add_feature(-1*np.ones(dim_features))


transitions = []

for i in range(nclusters + 2):
    transitions.append([])
    for j in range(nActions):
        transitions[i].append([])
        for k in range(nclusters + 2):
            transitions[i][j].append(0)

tot_traj = 0
for i in range(ntraj):
    j = 0
    trajectory = []
    while jump*j < traj_len[i]:
        trajectory.append(Kmlabels[tot_traj+j])
        j += 1
    tot_traj += int(traj_len[i]/jump)

    # add final state feature expectations (mortality):
    if r[i] == 1:
        trajectory.append(nclusters + 1)
    else:
        trajectory.append(nclusters)

    k = 0
    while jump*k < traj_len[i]:
        transitions[trajectory[k]][a[i][k*jump][0]-1][trajectory[k+1]] += 1
        k += 1


THETA = Transitions(num_states=nclusters+3,num_actions=nActions)

for i in range(nclusters):
    for j in range(nActions):
        sum_trans = 0
        for k in range(nclusters + 2):
            sum_trans += transitions[i][j][k]
        if (sum_trans == 0):
            THETA.set_trans(i,j,nclusters + 2,1)
        else:
            for k in range(nclusters + 2):
                prob = float(transitions[i][j][k])/float(sum_trans)
                THETA.set_trans(i,j,k,prob)
for i in range(nActions):
    THETA.set_trans(nclusters,i,nclusters,1)
    THETA.set_trans(nclusters + 1,i,nclusters + 1,1)
    THETA.set_trans(nclusters + 2,i,nclusters + 2,1)


mdp = ICMDP()
mdp.set_THETA(THETA)
mdp.set_F(F)

#######################################################################################################################
# Test the ellipsoid method in the environment
#######################################################################################################################
d={}

# load test and train sets of contexts and initial states:
testset = np.load("../../data/testset.npy")
test_init_states = np.load("../../data/test_init_states.npy")
train_contexts = np.load("../../data/trainset.npy")
train_init_states = np.load("../../data/train_init_states.npy")

# Evaluate expert on test set:
if RUN_TEST:
    policies_expert = []
    test_expert_value = []
    for jk in range(len(test_init_states)):
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W, context=testset[jk], flag='init',init_state=test_init_states[jk])
        policies_expert.append(features_expert.policy)
        value_expert = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value
train_rand_inds = np.arange(len(train_init_states))

# run seeds:
for trainset in range(repeats):
    save_obj(d, "values"+str(valuesindex))

    # random subset of the trainset:
    random.shuffle(train_rand_inds)
    Conts = train_contexts[train_rand_inds].copy()
    Conts = Conts[:iters]

    train_inits = train_init_states[train_rand_inds].copy()
    train_inits = train_inits[:iters]

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cumm_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0
    ellipsoid_volume = np.zeros(len(Conts))
    ERR = False

    E = Ellipsoid(real_W.size)

    max_updates = 2*real_W.size*(real_W.size + 1)*np.log(4*np.sqrt(real_W.size)/epsilon)
    min_ellipsoid_vol = (epsilon/2)**real_W.size
    num_seen = 0
    accuracy = []
    action_dist = []

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",context_count)

        # Agent and teacher play:
        Wt = E.getc().reshape(real_W.shape)
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W,context=Conts[t],flag='init',init_state=train_inits[t])
        features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol,  W=Wt, context=Conts[t],flag='init',init_state=train_inits[t])
        value_expert = ((1-gamma)/real_W.shape[1]) * Conts[t] @ real_W @ features_expert.M
        value_agent = ((1-gamma)/real_W.shape[1]) * Conts[t] @ real_W @ features_agent.M

        # Sanity checks:
        if t > 0 and ellipsoid_volume[t] > ellipsoid_volume[t-1]:
            print("ERR: Volume increased. t=",t," trainset=",trainset)
            ERR = True
        if context_count > max_updates:
            print("ERR: Went over max updates!. t=",t," trainset=",trainset)
            ERR = True
        if E.volume() < min_ellipsoid_vol:
            print("ERR: Ellipsoid too small!. t=",t," trainset=",trainset)
            ERR = True
        if not E.inside(real_W.reshape(real_W.size)):
            print("ERR: Real solution no longer in ellipsoid. t=",t," trainset=",trainset)
            ERR = True

        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count
        ellipsoid_volume[t] = E.volume()
        if t>0:
            cumm_regret[t] = cumm_regret[t-1] + value_expert - value_agent
        elif t==0:
            cumm_regret[t] = value_expert - value_agent

        # Calculate values on test set:
        # run test once every 10 contexts:
        if ( t > 0 and contexts_seen[t] != contexts_seen[t-1]):
            num_seen += 1

        if (RUN_TEST and num_seen >= 10) or (RUN_TEST and t==0):
            num_seen = 0
            test_agent_value = []
            accur = np.zeros(len(test_init_states))
            act_dist = np.zeros(len(test_init_states))

            for jk in range(len(test_init_states)):
                features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol, W=Wt, context=testset[jk], flag='init',init_state=test_init_states[jk])
                value_agent = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_agent.M
                test_agent_value.append(value_agent)
                accur[jk] = accuracy_mesure(features_agent.policy,policies_expert[jk])
                act_dist[jk] = actions_distance(features_agent.policy,policies_expert[jk])

            accuracy.append(accur.mean())
            action_dist.append(act_dist.mean())

            test_agent_value = np.asarray(test_agent_value).mean()
            print("test expert value: ",test_expert_value)
            print("test agent value: ", test_agent_value)
            d[trainset,"test_value",contexts_seen[t]] = test_agent_value

        # If agent is more than epsilon suboptimal, update ellipsoid:
        if value_expert - value_agent > epsilon:
            E.update(np.outer(Conts[t],features_expert.M-features_agent.M).reshape(real_W.size))
            context_count += 1


    # save data:
    np.save("obj/accuracy_" + str(trainset) + ".npy",np.asarray(accuracy))
    np.save("obj/action_distance_" + str(trainset) + ".npy",np.asarray(action_dist))
    np.save("obj/W_" + str(trainset) + ".npy",Wt)
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen
    d[trainset,"cumm_regret"] = cumm_regret

save_obj(d, "values"+str(valuesindex))
