#!/usr/bin/python3
# coding=utf8

import sys
import cv2
import time
import math
import threading
import pickle

import numpy as np
from enum import Enum
from pug_sdk import misc as Misc
from pug_sdk import yaml_handle

from collections import deque
import random

HomePath = '/home/hiwonder'
sys.path.append(HomePath) 
import Camera 

sys.path.append(HomePath + '/Pug_PC_Software') 
from ServoCmd import *
from ActionGroupControl import runAction, stopActionGroup
from HiwonderPuppy import HiwonderPuppy, BusServoParams

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# Robot movement configurations
Debug = 1
img_centerx = 160
PROCESS = 'find_ball'

# Ball detection variables
ball_centerx, ball_centery = -1, -1
ball_radius = 0

puppy = HiwonderPuppy(setServoPulse=setServoPulse, servoParams=BusServoParams(), dof='12')
Stand = {'roll':math.radians(0), 'pitch':math.radians(0), 'yaw':0.000, 'height':-13, 'x_shift':0.4, 'stance_x':1, 'stance_y':0}
Bend = {'roll':math.radians(0), 'pitch':math.radians(-21), 'yaw':0.000, 'height':-13, 'x_shift':0.4, 'stance_x':1, 'stance_y':0}

PuppyPose = Bend.copy()
GaitConfig = {'overlap_time':0.2, 'swing_time':0.15, 'clearance_time':0.0, 'z_clearance':6}
PuppyMove = {'x': 0, 'y': 0, 'yaw_rate': 0}

# RL parameters
ACTIONS = ['forward', 'move_left', 'move_right']
STATE_SIZE = 2  # ball_centerx, ball_centery
ACTION_SIZE = len(ACTIONS)
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MEMORY_SIZE = 1000
BATCH_SIZE = 32
LOAD_MODEL = True
GOAL_DONE = False
ITERATIONS = 5

# Experience replay memory
memory = deque(maxlen=MEMORY_SIZE)

def stance(x = 0, y = 0, z = -13, x_shift = 0):# Unit CM
    # The smaller the x_shift is, the more it leans forward when walking, and the larger it is, the more it leans back. 
    # By adjusting x_shift, the balance of the puppy's walking can be adjusted.
    return np.array([
                        [x + x_shift, x + x_shift, -x + x_shift, -x + x_shift],
                        [y, y, y, y],
                        [z, z, z, z],
                    ])#Do not change the combination of this array

# Initial stance position
def initMove():
    PuppyPose = Stand.copy()
    puppy.stance_config(stance(PuppyPose['stance_x'], PuppyPose['stance_y'], PuppyPose['height'], PuppyPose['x_shift']), PuppyPose['pitch'], PuppyPose['roll'])
    time.sleep(0.5)

# Stand position
def stand_move():
    PuppyPose = Stand.copy()
    puppy.stance_config(stance(PuppyPose['stance_x'], PuppyPose['stance_y'], PuppyPose['height'], PuppyPose['x_shift']), PuppyPose['pitch'], PuppyPose['roll'])
    time.sleep(0.5)

# Find ball position (Bend down)
def find_ball_move():
    PuppyPose = Bend.copy()
    puppy.stance_config(stance(PuppyPose['stance_x'], PuppyPose['stance_y'], PuppyPose['height'], PuppyPose['x_shift']), PuppyPose['pitch'], PuppyPose['roll'])
    time.sleep(0.2)

def load_config():
    global lab_data
    lab_data = yaml_handle.get_yaml_data('/home/hiwonder/pug/src/lab_config/config/lab_config.yaml')['color_range_list']

def init():
    puppy.gait_config(overlap_time=GaitConfig['overlap_time'], swing_time=GaitConfig['swing_time'], clearance_time=GaitConfig['clearance_time'], z_clearance=GaitConfig['z_clearance'])
    puppy.start()  # Start Up
    puppy.move_stop(servo_run_time=500)
    load_config()
    initMove()

# Movement functions
def forward(speed, t):
    PuppyMove['x'] = speed
    puppy.move(x=PuppyMove['x'], y=0, yaw_rate=0)
    time.sleep(t)

def move_left(speed, t):
    PuppyMove['y'] = speed
    puppy.move(x=0, y=PuppyMove['y'], yaw_rate=0)
    time.sleep(t)

def move_right(speed, t):
    PuppyMove['y'] = -speed
    puppy.move(x=0, y=PuppyMove['y'], yaw_rate=0)
    time.sleep(t)

def stop_move(t):
    puppy.move(x=0, y=0, yaw_rate=0)
    puppy.move_stop(servo_run_time=500)
    time.sleep(t)

# Reinforcement learning
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        self.q_table = {}

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(ACTIONS)
        # Get the action with the highest Q-value for the current state
        q_values = self.q_table.get(tuple(state), np.zeros(len(ACTIONS)))
        return ACTIONS[np.argmax(q_values)]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            # Convert state and next_state to tuples to use them as keys in q_table
            state = tuple(state)
            next_state = tuple(next_state)

            # Initialize Q-values for unseen states
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(ACTIONS))
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(len(ACTIONS))

            # Q-value update for the chosen action
            action_index = ACTIONS.index(action)

            # Compute the target: reward + discounted future Q-value
            if done:
                target = reward  # No future reward if the episode is done
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state])

            # Q-learning update rule: update the Q-value for state-action pair
            self.q_table[state][action_index] += self.learning_rate * (target - self.q_table[state][action_index])

        # Decrease exploration over time
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print('=====================================> Model SAVED to', filename)

    def load(self, filename):
        # if file doesnt exist, return an empty q_table
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print('=====================================> Model LOADED from', filename)
        except FileNotFoundError:
            self.q_table = {}
            print('Model file not found. Starting with a new Q-table.')
            

agent = DQNAgent()
if LOAD_MODEL:
    agent.load('dqn_model.pkl')

# Calculate reward based on the ball's position 
# Criteria:
# 1. The ball should be in the bottom right corner of the image (so that the robot can kick it with its right leg)
# 2. The ball should be closer to the bottom of the image (so that the robot can kick it with its foot)
def calculate_reward(ball_centerx, ball_centery):
    reward = 0
    x_dist = ball_centerx - 260 # Distance from the ball to the right edge of the image
    if ball_centery < 150: # Ball is in the top half of the image
        reward -= 1
    if x_dist in range(-20, 9): # Ball in a certain range from the right edge of the image
        reward += 1
    if ball_centery in range(220, 240): # Ball is in the bottom of the image (close to the robot's foot)
        reward += 1
    # reward is maximum when the ball is in the bottom right corner of the image
    return reward

def goal_reward():
    # Wait for user input to determine if the robot kicked the ball
    # input will be '1' if the robot kicked the ball, and any other input will be considered as '0' (robot didn't kick the ball)
    global GOAL_DONE
    GOAL_DONE = True
    user_input = input("Did the robot kick the ball? (1/0): ").strip()
    if user_input == '1':
        return 10
    else:
        return 0 # No reward if the robot didn't kick the ball
    
# Main movement function using RL
def move():
    global PROCESS, GOAL_DONE, ITERATIONS
    global ball_centerx, ball_centery
    time.sleep(5)
    while True:
        if PROCESS == 'find_ball':
            state = [ball_centerx, ball_centery]
            action = agent.choose_action(state)
            if action == 'forward':
                forward(10, 0.2)
            elif action == 'move_left':
                move_left(3, 0.2)
            elif action == 'move_right':
                move_right(3, 0.2)

            # Assume you have a way to calculate reward based on the ball's position
            reward = calculate_reward(ball_centerx, ball_centery)
            next_state = [ball_centerx, ball_centery]
            done = (PROCESS == 'kick_ball')  # or some condition to end the episode
                
            agent.remember(state, action, reward, next_state, done)
            agent.replay()  # Train the agent

            if ball_centery >= 220:
                PROCESS = 'kick_ball'
                

        elif PROCESS == 'kick_ball':
            stop_move(0.5)
            stand_move()
            runAction('kick_ball_right_2.d6ac')
            time.sleep(1) # Wait for the kick action to complete

            # Manually input the reward based on the robot's performance on kicking the ball
            if not GOAL_DONE: # We have to check global variable GOAL_DONE to avoid asking for input multiple times
                reward = goal_reward()
                state = [ball_centerx, ball_centery]
                next_state = [ball_centerx, ball_centery]
                agent.remember(state, action, reward, next_state, True) # Giving the final reward to the final action since we are not giving the robot ability to kick by itself rather we kick when certain conditions are met
                agent.replay()
                
            # Save the model after each episode
            agent.save('dqn_model.pkl')
            PROCESS = 'End'

        


# Run sub-threads to control robot movement
th = threading.Thread(target=move)
th.setDaemon(True)
# Print necessary information regarding thread
if Debug == 1:
    th.start()

# Main thread for image processing
def run(img):
    global PROCESS, ITERATIONS
    if PROCESS == 'End':
        return img
    if PROCESS == 'find_ball' or PROCESS == 'kick_ball':
        img = ball_detect(img, 'green')
    if PROCESS == 'Wait':
        img = cv2.putText(img, 'Wait for user input...', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if ITERATIONS > 0:
            ITERATIONS -= 1
            if input("Press Enter to continue...") == '':
                PROCESS = 'find_ball'
        else:
            print('End of the program')
            PROCESS = 'End'
    return img

def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None
    for c in contours:  # Go through all contours
        contour_area_temp = math.fabs(cv2.contourArea(c))  # Calculate the contour area
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp >= 5:  # Only when the area of the contour is greater than 5, the contour is considered effective
                area_max_contour = c
    return area_max_contour, contour_area_max  # Return the largest contour and its area

# Detect ball
def ball_detect(img, target_color='green'):
    global ball_centerx, ball_centery
    img_copy = img.copy()
    img_resize = cv2.resize(img_copy, (320, 240), interpolation=cv2.INTER_CUBIC)
    GaussianBlur_img = cv2.GaussianBlur(img_resize, (3, 3), 3)
    frame_lab = cv2.cvtColor(GaussianBlur_img, cv2.COLOR_BGR2LAB)
    frame_mask = cv2.inRange(frame_lab, (lab_data[target_color]['min'][0], lab_data[target_color]['min'][1], lab_data[target_color]['min'][2]),
                             (lab_data[target_color]['max'][0], lab_data[target_color]['max'][1], lab_data[target_color]['max'][2]))
    contours = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    areaMax_contour = getAreaMaxContour(contours)[0]
    if areaMax_contour is not None:
        ball = cv2.minEnclosingCircle(areaMax_contour)
        ball_centerx, ball_centery = int(ball[0][0]), int(ball[0][1])
        cv2.circle(img, (ball_centerx, ball_centery), int(ball[1]), (255, 0, 0), 2)
        cv2.circle(img, (ball_centerx, ball_centery), 2, (0, 255, 0), 2)
    else:
        ball_centerx, ball_centery = -1, -1
    return img

# 320x240 image resolution
#
#  0,0________320,0
#   |           |
#   |           |
#  240,0______320,240
#

# Main program loop
if __name__ == '__main__':
    init()
    find_ball_move()
    my_camera = Camera.Camera()
    my_camera.camera_open()
    while True:
        img, frame = my_camera.cap.read()
        if img is not None:
            frame1 = cv2.resize(frame, (320, 240))
            Frame = run(frame1)
            cv2.imshow('Frame', Frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
    my_camera.camera_close()
    stop_move(0.5)
    cv2.destroyAllWindows()