#######################################################################################################################
#   file name: bbl2_driving
#
#   description:
#   this file builds a driving simulator environment.
#   then runs a blackbox solver with the 2nd loss to solve the linear CIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../imports')
import numpy as np
import os
from ICMDP import *
import pickle
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
#HYPER PARAMS
#######################################################################################################################
gamma = 0.9
iters = 500
epsilon = 1e-3
repeats = 10
tol = 1e-4
test_size = 80
RUN_TEST = True
#######################################################################################################################
# construct the CMDP
#######################################################################################################################
# returns the new speed and position given  previous state, action and the environment parameters:
def do_action(state,action, min_speed, max_speed, min_x, max_x, step_size):
    speed = state.speed
    my_x = state.x
    
    # move left:
    if action == 1:
        if my_x - step_size >= min_x:
            my_x = my_x - step_size
        else:
            my_x = min_x
    
    # move right:
    elif action == 2:
        if my_x + step_size <= max_x:
            my_x = my_x + step_size
        else:
            my_x = max_x
    
    # increase speed:
    elif action == 3:
        if speed < max_speed:
            speed = speed + 1
    
    # decrease speed:
    elif action == 4:
        if speed > min_speed:
            speed = speed - 1

    return [speed,my_x]

# find next state function - finds the next possible states for a given state and action:
def find_next_state(state,action,states_inv):
    [new_speed, new_x] = do_action(state,action,speeds_num[0],speeds_num[-1], left_bound,right_bound,5)

    # check if this is the first state:
    if (states_inv[','.join(str(elem) for elem in (state.as_list()))] == 0):

        # for first state - special control for speed, to choose the speed for the rest of the game:
        state_vec = []
        for x in other_car_x:
            if action == 0:
                init_speed = state.speed
            elif action == 1:
                init_speed = state.speed - 1
            elif action == 2:
                init_speed = state.speed + 1

            # insert a car in a random position:
            new_state = states_inv[ ','.join(str(elem) for elem in [init_speed,state.x,x,10])]
            state_vec.append(new_state)
        return state_vec

    # check if need to insert a new car in a random place and remove the old one:
    elif (state.other_car[1] + displace[state.speed] >= height - 10 + my_car_size[0]):
        state_vec = []
        for x in other_car_x:
            new_state = states_inv[ ','.join(str(elem) for elem in [new_speed,new_x,x,10])]
            state_vec.append(new_state)
        return state_vec

    # no new car needed - deterministic next state:
    else:
        new_state = states_inv[','.join(str(elem) for elem in [new_speed, new_x ,state.other_car[0],state.other_car[1] + displace[state.speed]])]
        return new_state


# define parameters of the environment:
# features:
# 1. speed
# 2. collisions
# 3. off-road

# actions:
# 0 - do nothing
# 1 -  move left
# 2 - move right

# parameters:
# right-left step size:
step_size = 5

# boundaries of the frame
left_bound = 120
right_bound = 200
height = 180
width = 300
bottom_bound = height

# boundaries of the road:
road_left_bound = left_bound + 20
road_right_bound = right_bound - 20

# car size, width is half of the width in the format "[length,width]":
my_car_size = [40, 10]

# the y position of the player's car (stays fixed during the game):
my_y = height - 10 - my_car_size[0]

# initiate the speed feature values, displace for each speed and numbering:
displace = [20, 40, 80]
speeds_num = [0,1,2]
speed_feature_vals = [0.5,0.75,1]

# calculate the different possible x positions of the player's car:
my_x = []
for x in range(left_bound,right_bound + step_size,step_size):
    my_x.append(x)

# the lanes locations:
lanes = [140,160,180] # the x coordinates of the lanes

# build other_car:
other_car_length = 40
other_car_width = 10
other_car_x = lanes # to lower complexity
other_car_y = [] # the legal y coordinates of the other cars
for i in range(10):
    other_car_y.append(20*i + 10)

other_car = [] # format: [x coordinate, y coordinate]
for x in other_car_x:
    for y in other_car_y:
        other_car.append([x,y])

# build actions:
# 0 - do nothing
# 1 - move left
# 2 - move right
actions = [0,1,2]

# initiate states array and state to index (states_inv) dictionary:
states = []
states_inv = {}

# initiate features:
F = Features(dim_features=3)

# add first  state:
states.append(State(1,160,[-1,-1]))
states_inv[','.join(str(elem) for elem in (states[0].as_list()))] = 0
F.add_feature(feature=[0.75,0.5,0.5])

# build the whole state - feature mapping:
for speed in speeds_num:
    for x in my_x:
        for other_x in other_car_x:
            for other_y in other_car_y:
                states.append(State(speed,x,[other_x,other_y]))
                states_inv[','.join(str(elem) for elem in (states[len(states)-1].as_list()))] = len(states) - 1

                # add speed feature value:
                speed_val = speed_feature_vals[speed]
                
                # check collision:
                if (other_y > my_y) and (other_y - other_car_length < my_y + my_car_size[0]) and (other_x + other_car_width > x - my_car_size[1]) and (other_x - other_car_width < x + my_car_size[1]):
                    collision_val = 0
                else:
                    collision_val = 0.5

                # check off-road:
                if (x < road_left_bound) or (x  > road_right_bound):
                    off_road_val = 0
                else:
                    off_road_val = 0.5

                F.add_feature(feature=[speed_val,collision_val,off_road_val])


# setup transitions:
THETA = Transitions(num_states=len(states), num_actions=len(actions))
curr_state = 0
for state in states:
    for action in actions:

        # find next state:
        new_state = find_next_state(state,action,states_inv)

        # if there is more than 1 possible next state, calculate uniform distribution between the possibilities:
        if isinstance(new_state, list):
            num_states = len(new_state)
            trans = 1.0/num_states
            for i in range(num_states):
                THETA.set_trans(curr_state,action,new_state[i],trans)
        
        # deterministic next state:
        else:
            THETA.set_trans(curr_state,action,new_state,1)

    curr_state = curr_state + 1

# initiate an ICMDP object:
mdp = ICMDP()

# set the calculated features and transitions:
mdp.set_F(F)
mdp.set_THETA(THETA)

# load real W:
real_W = np.load("../../data/realW.npy")

#######################################################################################################################
# define functions for the blackbox optimizer
#######################################################################################################################

def feat_exp(r):
    return mdp.solve_MDP(gamma=gamma,tol=tol,w=r,flag = 'init').M

#real_W = real_W/(abs(real_W).max())
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

def evaluate_verbose(eval_map, training_contexts_all, expert_feat_exp_all):
    agent_r_est = [NL_est(training_contexts_all[j],eval_map) for j in range(len(training_contexts_all))]
    agent_feat_exp = [feat_exp(NL_est(training_contexts_all[j],eval_map)) for j in range(len(training_contexts_all))]
    value_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j]))**2 \
                            for j in range(len(training_contexts_all))]).sum()
    feat_err  = np.asarray([np.linalg.norm(agent_feat_exp[j] - expert_feat_exp_all[j],2) \
                            for j in range(len(training_contexts_all))]).sum()
    dir_err = np.asarray([(agent_r_est[j] @ (agent_feat_exp[j] - expert_feat_exp_all[j])) \
                            for j in range(len(training_contexts_all))]).sum()
    print("value err: ",value_err, " feat_exp err: ",feat_err, " dir err: ",dir_err)

def Update_estimator(mat_list,training_contexts_all, expert_feat_exp_all, max_iter=500,
                     stepsize=0.05, decay=0.99, std=0.1, num_pts=20, batch_size=1, epsilon=1e-3,
                     debug_interval = 50):
    # Create copy to not affect input matrices
    curr_list = [mat.copy() for mat in mat_list]

    # Initialize probability vector, make sure it is updated in 1st iteration
    probs = np.zeros(len(training_contexts_all))

    iteration = 1
    while iteration <= max_iter:
        agent_feat_exp_all = np.array([feat_exp(NL_est(training_context,curr_list)) for training_context in training_contexts_all])
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
            test_agent_value_test = np.asarray([((1-gamma)/3) * NL(ctest) @ feat_exp(NL_est(ctest,curr_list)) for ctest in testset]).mean()
            test_agent_value_train = np.asarray([((1-gamma)/3) * NL(ctest) @ feat_exp(NL_est(ctest,curr_list)) for ctest in training_contexts_all]).mean()
            print("Agent value test:",test_agent_value_test)
            print("Agent value train:",test_agent_value_train)

            print("Train errors:")
            evaluate_verbose(curr_list, training_contexts_all, expert_feat_exp_all)
            print("Test errors:")
            evaluate_verbose(curr_list, testset, testfeat)

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
d = {}
    
test_expert_value = 0
# load test set of contexts:
testset = np.load("../../data/test_set.npy")[:test_size]

# Evaluate expert on test set
if RUN_TEST:
    test_expert_value = np.asarray([((1-gamma)/3) * NL(c) @ feat_exp(NL(c)) for c in testset]).mean()
    d["test_value"] = test_expert_value
    print("Expert test value: ",str(test_expert_value))

# run seeds:
for trainset in range(repeats):
    if trainset>0:
        save_obj(d, "values"+str(valuesindex))

    # load train set of contexts:
    Conts = np.load("../../data/train_set_"+str(trainset)+".npy")[:iters]

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cum_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0

    weights = []
    weights.append(np.random.normal(size=(3,3)).reshape(3,3))
    weights = [mat/np.linalg.norm(mat.flatten()) for mat in weights]
    training_contexts = []
    training_features = []

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",len(training_contexts))

        # Agent and teacher play:
        ct = Conts[t]
        r = NL(ct)
        r_est = NL_est(ct,weights)
        features_expert = feat_exp(r)
        features_agent = feat_exp(r_est)
        value_expert = ((1-gamma)/3) * r @ features_expert
        value_agent = ((1-gamma)/3) * r @ features_agent

        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count
        if t>0:
            cum_regret[t] = cum_regret[t-1] + value_expert - value_agent
        elif t==0:
            cum_regret[t] = value_expert - value_agent

        # Calculate values on test set:
        if (RUN_TEST and t>0 and contexts_seen[t] % 1 == 0 and contexts_seen[t] != contexts_seen[t-1]) or (RUN_TEST and t==0):
            test_agent_value = np.asarray([((1-gamma)/3) * NL(c) @ feat_exp(NL_est(c,weights)) for c in testset]).mean()
            d[trainset,"test_value",contexts_seen[t]] = test_agent_value
            print(" == Generalization for ",str(contexts_seen[t]), "contexts: ",str(test_agent_value))

        # If agent is more than epsilon suboptimal, update W:
        print("Value expert: ",str(value_expert)," Value agent: ",str(value_agent))
        if value_expert - value_agent > epsilon:
            training_contexts.append(ct)
            training_features.append(features_expert)
            print("Training with ",str(len(training_contexts)), " contexts:")
            weights = Update_estimator(weights, np.asarray(training_contexts),
                                       np.asarray(training_features), max_iter=50,
                                       stepsize=0.1, decay=0.95, std=0.001,
                                       num_pts=250, batch_size=len(training_contexts),
                                       epsilon=epsilon, debug_interval = 0)

            context_count += 1

    # save data:
    d[trainset,"expert_values"] = expert_values
    d[trainset,"agent_values"] = agent_values
    d[trainset,"contexts_seen"] = contexts_seen
    d[trainset,"cum_regret"] = cum_regret

save_obj(d, "values"+str(valuesindex))
