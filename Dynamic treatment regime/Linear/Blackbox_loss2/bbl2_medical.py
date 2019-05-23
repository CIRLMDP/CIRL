#######################################################################################################################
#   file name: bbl2_medical
#
#   description:
#   this file builds a medical environment from the MIMICIII normalized data.
#   then runs the blackbox method with the 2nd loss function to solve the linear CIRL problem.
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
epsilon = 5e-4
testi = 0
testf = 5
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

# add features of 2 possible final states:
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
# define functions for the blackbox optimizer
#######################################################################################################################
def feat_exp(r,init_state):
    return mdp.solve_MDP(gamma=0.9,tol=tol,w=r,flag = 'init',init_state=init_state)

def NL(x):
    return x @ real_W
    
def NL_est(x,mat_list):
    y = x
    for i in range(len(mat_list)):
        y = y @ mat_list[i]
    return y

def evaluate(map_eval, training_contexts, expert_feat_exp, agent_feat_exp):
    return np.asarray([(NL_est(training_contexts[j],map_eval) @ (agent_feat_exp[j] - expert_feat_exp[j])) \
                       for j in range(len(training_contexts))]).sum()

def evaluate_verbose(eval_map, training_contexts_all, expert_feat_exp_all, init_states_all):
    agent_r_est = [NL_est(training_contexts_all[j],eval_map) for j in range(len(training_contexts_all))]
    agent_feat_exp = [feat_exp(NL_est(training_contexts_all[j],eval_map),init_states_all[j]).M for j in range(len(training_contexts_all))]
    value_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j]))**2 \
                            for j in range(len(training_contexts_all))]).sum()
    feat_err  = np.asarray([np.linalg.norm(agent_feat_exp[j] - expert_feat_exp_all[j],2) \
                            for j in range(len(training_contexts_all))]).sum()
    dir_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j])) \
                            for j in range(len(training_contexts_all))]).sum()
    print("value err: ",value_err, " feat_exp err: ",feat_err, " dir err: ",dir_err)

def Update_estimator(mat_list,training_contexts_all, expert_feat_exp_all, init_states_all, max_iter=500,
                     stepsize=0.05, decay=0.99, std=0.1, num_pts=20, batch_size=1, epsilon=1e-3,
                     debug_interval = 50):
    # Create copy to not affect input matrices
    curr_list = [mat.copy() for mat in mat_list]

    # Initialize probability vector, make sure it is updated in 1st iteration
    probs = np.zeros(len(training_contexts_all))

    iteration = 1
    while iteration <= max_iter:
        agent_feat_exp_all = np.array([feat_exp(NL_est(training_contexts_all[j],curr_list), init_states_all[j]).M for j in range(len(training_contexts_all))])
        # early stopping condition
        if np.array([np.linalg.norm(agent_feat_exp_all[j]-expert_feat_exp_all[j],1) for j in range(len(expert_feat_exp_all))]).max() < epsilon:
            print("Early stop iteration ",iteration)
            return curr_list
        
        probs = np.array([evaluate(curr_list, np.expand_dims(training_contexts_all[j],0),
                          np.expand_dims(expert_feat_exp_all[j],0), np.expand_dims(agent_feat_exp_all[j],0)) for j in range(len(training_contexts_all))])

        # Select minibatch for this iteration using the probability vector
        probs = np.maximum(probs,0)
        probs += 1e-4
        indices = np.random.choice(a=list(range(len(training_contexts_all))), p=probs/probs.sum(), size=batch_size, replace=False)
        training_contexts = training_contexts_all[indices]
        expert_feat_exp = expert_feat_exp_all[indices]
        agent_feat_exp = agent_feat_exp_all[indices]

        # Evaluate starting point for this minibatch. If minibatch is already close to optimal, go back and try again
        ev = evaluate(curr_list, training_contexts, expert_feat_exp, agent_feat_exp)
        
        #DEBUG PRINT
        if debug_interval:
            print("Iter ",iteration," out of ",max_iter)
        if debug_interval and iteration % debug_interval == 0:
            test_agent_value_test = np.asarray([((1-gamma)/real_W.shape[1]) * NL(testset[j]) @ feat_exp(NL_est(testset[j],curr_list),test_init_states[j]).M for j in range(len(testset))]).mean()
            test_agent_value_train = np.asarray([((1-gamma)/real_W.shape[1]) * NL(training_contexts_all[j]) @ feat_exp(NL_est(training_contexts_all[j],curr_list),init_states_all[j]).M for j in range(len(training_contexts_all))]).mean()
            print("Agent value test:",test_agent_value_test)
            print("Agent value train:",test_agent_value_train)

            print("Train errors:")
            evaluate_verbose(curr_list, training_contexts_all, expert_feat_exp_all,init_states_all)
            print("Test errors:")
            evaluate_verbose(curr_list, testset, testfeat, test_init_states)

        # Initialize step, calculate std for noise with decay
        mat_num = len(curr_list)
        step = [np.zeros(shape=mat.shape) for mat in curr_list]
        for _ in range(int(num_pts/2)):
            # Create gaussian noises for matrices
            noise = [np.random.normal(size=mat.shape) for mat in curr_list]
            # Calculate step in both directions
            step_plus = [curr_list[j] + std*noise[j] for j in range(mat_num)]
            step_minus = [curr_list[j] - std*noise[j] for j in range(mat_num)]
            # Evaluate function and update step
            eval_plus = evaluate(step_plus, training_contexts, expert_feat_exp, agent_feat_exp)
            eval_minus = evaluate(step_minus, training_contexts, expert_feat_exp, agent_feat_exp)
            step = [step[j] + (eval_plus - eval_minus)*noise[j] for j in range(mat_num)]

        # Normalize step size
        step = [step[j] / np.linalg.norm(step[j].flatten()) for j in range(mat_num)]
        # Update current point
        curr_list = [curr_list[j] - stepsize*(decay**iteration)*step[j] for j in range(mat_num)]
        curr_list = [mat/np.linalg.norm(mat.flatten()) for mat in curr_list]
        # If point is worse on this minibatch, cancel this step
        iteration += 1

    return curr_list

#######################################################################################################################
# Test the blackbox method in the environment
#######################################################################################################################
d={}

# load test and train sets of contexts and initial states:
testset = np.load("../../data/testset.npy")
test_init_states = np.load("../../data/test_init_states.npy")
train_contexts = np.load("../../data/trainset.npy")
train_init_states = np.load("../../data/train_init_states.npy")
testfeat = []

# Evaluate expert on test set:
if RUN_TEST:
    policies_expert = []
    test_expert_value = []
    for jk in range(len(test_init_states)):
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W, context=testset[jk], flag='init',init_state=test_init_states[jk])
        policies_expert.append(features_expert.policy)
        testfeat.append(features_expert.M)
        value_expert = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_expert.M
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

# run seeds:
testfeat = np.array(testfeat)
train_rand_inds = np.arange(len(train_init_states))
for trainset in range(testi,testf):
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
    ERR = False

    num_seen = 0
    accuracy = []
    action_dist = []
    
    # start from random W:
    weights = []
    weights.append(np.random.normal(size=(8,42)).reshape(8,42))
    weights = [mat/np.linalg.norm(mat.flatten()) for mat in weights]
    training_contexts = []
    training_features = []
    training_inits = []

    for t in range(iters):
        print("Run ",valuesindex," test ",trainset, " of ", str(testi), "->" , str(testf),"timestep ",t, " contexts seen ",context_count)

        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, W=real_W,context=Conts[t],flag='init',init_state=train_inits[t])
        r_est = NL_est(Conts[t],weights)
        features_agent = feat_exp(r_est, train_inits[t])
        value_expert = ((1-gamma)/real_W.shape[1]) * Conts[t] @ real_W @ features_expert.M
        value_agent = ((1-gamma)/real_W.shape[1]) * Conts[t] @ real_W @ features_agent.M
        
        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count
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
                r_est = NL_est(testset[jk],weights)
                features_agent = feat_exp(r_est, test_init_states[jk])
                value_agent = ((1-gamma)/real_W.shape[1]) * testset[jk] @ real_W @ features_agent.M
                test_agent_value.append(value_agent)
                accur[jk] = accuracy_mesure(features_agent.policy,policies_expert[jk])
                act_dist[jk] = actions_distance(features_agent.policy,policies_expert[jk])

            accuracy.append(accur.mean())
            action_dist.append(act_dist.mean())

            test_agent_value = np.asarray(test_agent_value).mean()
            d[trainset,"test_value",contexts_seen[t]] = test_agent_value
            print(" == Generalization for ",str(contexts_seen[t]), "contexts: ",str(test_agent_value))

        # If agent is more than epsilon suboptimal, update W using black box solver:
        if value_expert - value_agent > epsilon:
            training_contexts.append(Conts[t])
            training_inits.append(train_inits[t])
            training_features.append(features_expert.M)
            print("Training with ",str(len(training_contexts)), " contexts:")
            weights = Update_estimator(weights, np.asarray(training_contexts),
                                       np.asarray(training_features),
                                       np.array(training_inits),
                                       max_iter=80,
                                       stepsize=0.25, decay=0.95, std=0.0001,
                                       num_pts=1000, batch_size=len(training_contexts),
                                       epsilon=epsilon, debug_interval = 0)
            context_count += 1

    # save data:
    np.save("obj/accuracy_" + str(trainset) + ".npy",np.asarray(accuracy))
    np.save("obj/action_distance_" + str(trainset) + ".npy",np.asarray(action_dist))
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen
    d[trainset,"cumm_regret"] = cumm_regret

save_obj(d, "values"+str(valuesindex))
