#######################################################################################################################
#   file name: bb_driving
#
#   description:
#   this file builds a driving simulator environment
#   then runs the blackbox method to solve the linear CIRL problem.
#######################################################################################################################
# imports:
#######################################################################################################################
import sys
sys.path.append('../../imports')
import os
import numpy as np
from ICMDP import *
from ES_opt import *
import random
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
iters = 500
epsilon = 1e-3
gamma = 0.9
repeats = 10
tol = 1e-4
max_epochs = 50
num_eps = 8
init_step = 0.1
step_decay = 0.94
sigma = 0.1
sigma_decay = 1
test_size = 80
RUN_TEST = True
#######################################################################################################################
# construct the CMDP
#######################################################################################################################
# returns the new speed and position given  previous state, action and the environment parameters:
def do_action(state, action, min_speed, max_speed, min_x, max_x, step_size):
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

    return [speed, my_x]

# find next state function - finds the next possible states for a given state and action:
def find_next_state(state, action, states_inv):
    [new_speed, new_x] = do_action(state, action, speeds_num[0], speeds_num[-1], left_bound, right_bound, 5)

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
            new_state = states_inv[','.join(str(elem) for elem in [init_speed, state.x, x, 10])]
            state_vec.append(new_state)
        return state_vec

    # check if need to insert a new car in a random place and remove the old one:
    elif (state.other_car[1] + displace[state.speed] >= height - 10 + my_car_size[0]):
        state_vec = []
        for x in other_car_x:
            new_state = states_inv[','.join(str(elem) for elem in [new_speed, new_x, x, 10])]
            state_vec.append(new_state)
        return state_vec

    # no new car needed - deterministic next state:
    else:
        new_state = states_inv[','.join(
            str(elem) for elem in [new_speed, new_x, state.other_car[0], state.other_car[1] + displace[state.speed]])]
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
speeds_num = [0, 1, 2]
speed_feature_vals = [0.5, 0.75, 1]

# calculate the different possible x positions of the player's car:
my_x = []
for x in range(left_bound, right_bound + step_size, step_size):
    my_x.append(x)

# the lanes locations:
lanes = [140, 160, 180]  # the x coordinates of the lanes

# build other_car:
other_car_length = 40
other_car_width = 10
other_car_x = lanes  # to lower complexity
other_car_y = []  # the legal y coordinates of the other cars
for i in range(10):
    other_car_y.append(20 * i + 10)

other_car = []  # format: [x coordinate, y coordinate]
for x in other_car_x:
    for y in other_car_y:
        other_car.append([x, y])

# build actions:
# 0 - do nothing
# 1 - move left
# 2 - move right
actions = [0, 1, 2]

# initiate states array and state to index (states_inv) dictionary:
states = []
states_inv = {}

# initiate features:
F = Features(dim_features=3)

# add first  state:
states.append(State(1, 160, [-1, -1]))
states_inv[','.join(str(elem) for elem in (states[0].as_list()))] = 0
F.add_feature(feature=[0.75, 0.5, 0.5])

# build the whole state - feature mapping:
for speed in speeds_num:
    for x in my_x:
        for other_x in other_car_x:
            for other_y in other_car_y:
                states.append(State(speed, x, [other_x, other_y]))
                states_inv[','.join(str(elem) for elem in (states[len(states) - 1].as_list()))] = len(states) - 1

                # add speed feature value:
                speed_val = speed_feature_vals[speed]

                # check collision:
                if (other_y > my_y) and (other_y - other_car_length < my_y + my_car_size[0]) and (
                        other_x + other_car_width > x - my_car_size[1]) and (
                        other_x - other_car_width < x + my_car_size[1]):
                    collision_val = 0
                else:
                    collision_val = 0.5

                # check off-road:
                if (x < road_left_bound) or (x > road_right_bound):
                    off_road_val = 0
                else:
                    off_road_val = 0.5

                F.add_feature(feature=[speed_val, collision_val, off_road_val])

# setup transitions:
THETA = Transitions(num_states=len(states), num_actions=len(actions))
curr_state = 0
for state in states:
    for action in actions:

        # find next state:
        new_state = find_next_state(state, action, states_inv)

        # if there is more than 1 possible next state, calculate uniform distribution between the possibilities:
        if isinstance(new_state, list):
            num_states = len(new_state)
            trans = 1.0 / num_states
            for i in range(num_states):
                THETA.set_trans(curr_state, action, new_state[i], trans)

        # deterministic next state:
        else:
            THETA.set_trans(curr_state, action, new_state, 1)

    curr_state = curr_state + 1

# initiate an ICMDP object:
mdp = ICMDP()

# set the calculated features and transitions:
mdp.set_F(F)
mdp.set_THETA(THETA)

#######################################################################################################################
# Test the blackbox method in the environment
#######################################################################################################################
d = {}

real_W = np.load("../../data/realW.npy")
test_expert_value = 0
testset = np.load("../../data/test_set.npy")
testset = testset[:test_size]

# Evaluate expert on test set:
if RUN_TEST:
    test_expert_value = []
    for c in testset:
        mdp.set_C(c)
        mdp.set_W(real_W)
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
        value_expert = ((1 - gamma) / real_W.shape[1]) * np.matmul(c, np.matmul(real_W, features_expert.M))
        test_expert_value.append(value_expert)
    test_expert_value = np.asarray(test_expert_value).mean()
    d["test_value"] = test_expert_value

for trainset in range(repeats):
    Contexts = []
    expert_mus = []
    save_obj(d, "values"+str(valuesindex))

    Conts = np.load("../../data/train_set_" + str(trainset) + ".npy")
    Conts = Conts[:iters]

    expert_values = np.zeros(len(Conts))
    agent_values = np.zeros(len(Conts))
    cumm_regret = np.zeros(len(Conts))
    contexts_seen = np.zeros(len(Conts))
    context_count = 0
    ERR = False

    Wt = 2 * np.random.rand(3, 3) - 1
    Wt /= np.linalg.norm(Wt)

    for t in range(iters):
        print("test ",trainset," timestep ",t, " Contexts seen:",context_count)

        # Agent and teacher play:
        ct = Conts[t]
        mdp.set_C(ct)
        mdp.set_W(real_W)
        features_expert = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
        mdp.set_W(Wt)
        features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
        value_expert = ((1 - gamma) / real_W.shape[1]) * np.matmul(ct, np.matmul(real_W,features_expert.M))
        value_agent = ((1 - gamma) / real_W.shape[1]) * np.matmul(ct, np.matmul(real_W,features_agent.M))

        # Record results:
        expert_values[t] = value_expert
        agent_values[t] = value_agent
        contexts_seen[t] = context_count

        if t > 0:
            cumm_regret[t] = cumm_regret[t - 1] + value_expert - value_agent
        elif t == 0:
            cumm_regret[t] = value_expert - value_agent

        # Calculate values on test set:
        if (RUN_TEST and t > 0 and contexts_seen[t] % 5 == 0 and contexts_seen[t] != contexts_seen[t - 1]) or (
                RUN_TEST and t == 0):
            test_agent_value = []
            for c in testset:
                mdp.set_C(c)
                mdp.set_W(Wt)
                features_agent = mdp.solve_CMDP(gamma=gamma, tol=tol, flag='init')
                value_agent = ((1 - gamma) / real_W.shape[1]) * np.matmul(c, np.matmul(real_W, features_agent.M))
                test_agent_value.append(value_agent)
            test_agent_value = np.asarray(test_agent_value).mean()
            d[trainset, "test_value", contexts_seen[t]] = test_agent_value

        # If agent is more than epsilon suboptimal, update W:
        if (value_expert - value_agent > epsilon):
            print("context added, total of: ", len(Contexts) + 1)
            Contexts.append(ct)
            expert_mus.append(features_expert.M)

            ########################################################################################################################################
            # Blackbox algorithm
            ########################################################################################################################################

            step_size_opt = init_step
            sigma_opt = sigma
            rand_order = np.arange(len(Contexts))
            random.shuffle(rand_order)
            for mmm in range(max_epochs):
                done = True
                for k in rand_order:

                    # define the loss function for the current iteration:
                    func = lambda W:  mdp.feature_expectations_opt(W= W,gamma = gamma,contexts=[Contexts[k]],expert_mus=[expert_mus[k]],mode='value')

                    # run the black box optimizer:
                    res = ES_minimize(func, init_step=step_size_opt, sigma=sigma_opt, num_eps=num_eps, theta_init=Wt, maxiter=1,tol_stop=1e-4)
                    done = done and res.done
                    Wt = res.x

                step_size_opt *= step_decay
                sigma_opt *= sigma_decay

                if done:
                    break

            ########################################################################################################################################
            Wt = res.x
            Wt /= np.linalg.norm(Wt)

            context_count += 1

    # save data:
    d[trainset, "expert_values"] = expert_values
    d[trainset, "agent_values"] = agent_values
    d[trainset, "contexts_seen"] = contexts_seen
    d[trainset, "cumm_regret"] = cumm_regret

save_obj(d, "values"+str(valuesindex))
