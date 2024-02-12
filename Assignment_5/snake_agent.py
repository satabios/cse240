import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this
#   Credits to Chitian Hao for the help!
class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
 
    #   Function Helper:It gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   Returns a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        print("IN helper_func")
        head = [state[0],state[1]]
        head_x = head[0]
        head_y = head[1]
        body = state[2]
        food = [state[3], state[4]]
        food_x = food[0]
        food_y = food[1]
        body_list = []
        for i, j in body: body_list.append([i,j])
        wall_hit = [0,0]
        direction_of_food = [0,0]
        snake_body = []

        if head_x == 40: wall_hit[0] = 1
        elif head_x == 480: wall_hit[0] = 2
        else: wall_hit[0] = 0
        if head_y == 40: wall_hit[1] = 1
        elif head_y == 480: wall_hit[1] = 2
        else: wall_hit[1] = 0

        if (food_x - head_x) > 0: direction_of_food[0] = 2
        elif (food_x - head_x) < 0: direction_of_food[0] = 1
        else: direction_of_food[0] = 0
        if (food_y - head_y) > 0: direction_of_food[1] = 2
        elif (food_y - head_y) < 0: direction_of_food[1] = 1
        else: direction_of_food[1] = 0

        if [head_x, head_y-1] in body_list: collision = 1 
        else: collision = 0
        snake_body.append(collision)
        if [head_x, head_y+1] in body_list: collision = 1 
        else: collision = 0
        snake_body.append(collision)
        if [head_x-1, head_y] in body_list: collision = 1
        else: collision = 0
        snake_body.append(collision)
        if ((head_x+1, head_y) in body_list): collision = 1
        else: collision = 0
        snake_body.append(collision)
        condition_list = [wall_hit[0],wall_hit[1],direction_of_food[0],direction_of_food[1],snake_body[0],snake_body[1],snake_body[2],snake_body[3]]
        return condition_list

    # Computing the reward.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1


    def agent_action(self, state, points, dead):
        print("IN AGENT_ACTION")

        def change(state,dead,points,previous_state,previous_action):
            previous_move = self.helper_func(previous_state)
            r = self.compute_reward(points, dead)
            present_state = self.helper_func(state)
            upper = self.Q[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][0]
            bottom = self.Q[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][1]
            left = self.Q[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][2]
            right = self.Q[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][3]
            action = max(upper, bottom, left, right)
            q = self.Q[previous_move[0]][previous_move[1]][previous_move[2]][previous_move[3]][previous_move[4]][previous_move[5]][previous_move[6]][previous_move[7]][previous_action]
            alpha = self.LPC / (self.LPC + self.N[previous_move[0]][previous_move[1]][previous_move[2]][previous_move[3]]
                          [previous_move[4]][previous_move[5]][previous_move[6]][previous_move[7]][previous_action])
            change = q + alpha * (r + self.gamma * action - q)
            return change
        Qvalues = [0, 0, 0, 0]
        if dead:
            previous_state = self.helper_func(self.s)
            self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a] = change(state, dead, points, self.s, self.a)
            self.reset()
            return
        present_state = self.helper_func(state)
        if self._train and self.s != None and self.a != None:
            previous_state = self.helper_func(self.s)
            new_q = change(state, dead, points, self.s, self.a)
            self.Q[previous_state[0]][previous_state[1]][previous_state[2]][previous_state[3]][previous_state[4]][previous_state[5]][previous_state[6]][previous_state[7]][self.a] = new_q
        for i in range(helper.NUM_ACTIONS):
            n = self.N[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][i]
            q = self.Q[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][i]
            if n < self.Ne:
                Qvalues[i] = 1
            else:
                Qvalues[i] = q
        action = np.argmax(Qvalues)
        max_action = max(Qvalues)
        for i in range(len(Qvalues)-1, -1, -1):
            if Qvalues[i] == max_action:
                action = i
                break
        self.N[present_state[0]][present_state[1]][present_state[2]][present_state[3]][present_state[4]][present_state[5]][present_state[6]][present_state[7]][action] += 1
        self.s = state
        self.a = action
        self.points = points
        return action