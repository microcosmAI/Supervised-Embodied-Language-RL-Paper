from MuJoCo_Gym.mujoco_rl import MuJoCoRL
import numpy as np
import cv2
import copy
from sklearn.metrics import mean_squared_error
from autoencoder import Autoencoder
from ray.air import session
import math
import mujoco

class Image:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [0 for _ in range(50)], "high": [1 for _ in range(50)]}
        self.action_space = {"low": [], "high": []}
        self.autoencoder = Autoencoder(latent_dim=50, input_shape=(64, 64, 3))
        self.autoencoder.encoder.load_weights("models/encoder50.h5")
        self.index = 0

    def dynamic(self, agent, actions):
        self.index = self.index + 1
        image = self.environment.get_camera_data(agent + "_camera")
        image = cv2.resize(image, (64, 64))
        result = self.autoencoder.encoder.predict(np.array([image]), verbose=0)[0]
        # cv2.imwrite(f"/Users/cowolff/Documents/GitHub/s.language_experiments/images/{agent}_{self.index}.png", image)
        return 0, result, False, {}
    
class Communication:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [0, 0, 0, 0], "high": [1, 1, 1, 1]}
        self.action_space = {"low": [0, 0, 0, 0], "high": [1, 1, 1, 1]}

    def dynamic(self, agent, actions):
        if "utterance" not in self.environment.data_store[agent].keys():
            self.environment.data_store[agent]["utterance"] = None
        if agent == "receiver":
            utterance = [0, 0, 0, 0]
            if "target_color" in self.environment.data_store.keys():
                utterance[np.argmax(self.environment.data_store["target_color"])] = 1
            observation = utterance
            # if "utterance_max" in self.environment.data_store["sender"].keys():
            #     observation = self.environment.data_store["sender"]["utterance_max"]
            # else:
            #     observation = [0, 0, 0, 0]
            reward = 0
        elif agent == "sender":
            utterance = [0, 0, 0, 0]
            utterance[np.argmax(actions)] = 1
            self.environment.data_store[agent]["utterance"] = actions
            self.environment.data_store[agent]["utterance_max"] = utterance
            observation = [0, 0, 0, 0]

            # reference = [0, 0, 0, 0]
            # color = self.environment.data_store["target_color"]
            # reference[np.argmax(color)] = 1
            # reward = -1 * mean_squared_error(reference, actions)
        else:
            print("Dafaq is going on here?")
        return 0, observation, False, {}
    
class Accuracy:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [], "high": []}
        self.action_space = {"low": [], "high": []}
        self.accuracies = []
        self.variances = []
        self.sendAccuracies = []
        self.sendVariances = []
        self.currentSend = []
        self.report_accuracy = {"sender": 0, "receiver": 0}

    def dynamic(self, agent, actions):
        choices = ["choice_1", "choice_2"]
        variance = {"choice_1":1, "choice_2":-1}
        if "target" in self.environment.data_store.keys():
            if "sendVariances" not in self.environment.data_store.keys():
                self.environment.data_store["sendVariances"] = True
                self.currentSend = [0, 0, 0, 0]
            target = self.environment.data_store["target"]
            # if any(self.environment.collision(ankle, target + "_geom") for ankle in ["left_leg_geom_2", "left_ankle_geom_2", "right_leg_geom_2", "right_ankle_geom_2", "back_leg_geom_2", "third_ankle_geom_2", "rightback_leg_geom_2", "fourth_ankle_geom_2"]):
            if self.environment.collision("receiver_geom", target + "_geom"):
                self.accuracies.append(1)
                self.variances.append(variance[target])

                if len(self.variances) > 50:
                    report_variance = 1 - abs(sum(self.variances[-50:]) / 50)
                    self.report_accuracy["receiver"] = sum(self.accuracies[-50:]) / 50
            # elif any(self.environment.collision(ankle, [choice for choice in choices if choice != target][0] + "_geom") for ankle in ["left_leg_geom_2", "left_ankle_geom_2", "right_leg_geom_2", "right_ankle_geom_2", "back_leg_geom_2", "third_ankle_geom_2", "rightback_leg_geom_2", "fourth_ankle_geom_2"]):
            elif self.environment.collision("receiver_geom", [choice for choice in choices if choice != target][0] + "_geom"):
                self.accuracies.append(0)
                self.variances.append(variance[[choice for choice in choices if choice != target][0]])

                if len(self.variances) > 50:
                    report_variance = 1 - abs(sum(self.variances[-50:]) / 50)
                    self.report_accuracy[0] = sum(self.accuracies[-50:]) / 50
            if "utterance_max" in self.environment.data_store[agent].keys():
                reference = [0, 0, 0, 0]
                color = self.environment.data_store["target_color"]
                reference[np.argmax(color)] = 1
                self.currentSend = np.add(self.currentSend, self.environment.data_store[agent]["utterance_max"])

                if self.environment.data_store[agent]["utterance_max"]  == reference:
                    self.sendAccuracies.append(1)
                else:
                    self.sendAccuracies.append(0)
        return 0, [], False, {}
    
class Reward:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [], "high": []}
        self.action_space = {"low": [], "high": []}
        self.choices = ["choice_1", "choice_2"]

    def dynamic(self, agent, actions):
        if not "target" in self.environment.data_store.keys():
            color = self.environment.get_data("reference_geom")["color"]
            for choice in self.choices:
                if (color == self.environment.get_data(choice + "_geom")["color"]).all():
                    self.environment.data_store["target"] = choice
                    self.environment.data_store["target_color"] = self.environment.get_data(choice + "_geom")["color"]
                    self.environment.data_store["last_distance"] = copy.deepcopy(self.environment.distance("receiver_geom", choice + "_geom"))
        if agent == "receiver":
            target = self.environment.data_store["target"]
            new_distance = self.environment.distance("receiver_geom", target + "_geom")
            reward = (self.environment.data_store["last_distance"] - new_distance) * 10
            self.environment.data_store["last_distance"] = copy.deepcopy(new_distance)
        elif agent == "sender":
            reference = [0, 0, 0, 0]
            color = self.environment.data_store["target_color"]
            reference[np.argmax(color)] = 1
            reward = 0
            if "utterance" in self.environment.data_store[agent].keys():
                reward = -1 * mean_squared_error(reference, self.environment.data_store[agent]["utterance"])
        return reward, [], False, {}
    
class RayDynamic:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [0, 0, 0, 0], "high": [255, 255, 255, 255]}
        self.action_space = {"low": [], "high": []}
        self.senderId = np.array([self.environment.get_data("sender_geom")["id"]], np.int32)

    def dynamic(self, agent, action):
        geomId = copy.deepcopy(self.senderId)
        position = self.environment.data.site("sender_site").xpos
        rvec = [0, 0, 0]
        mat = self.environment.data.site("sender_site").xmat
        rvec[0] = mat[2]
        rvec[1] = mat[5]
        rvec[2] = mat[8]
        if sum(rvec) == 0:
            rvec[0] = 0.1
        result =  mujoco.mj_ray(self.environment.model, self.environment.data, position, rvec, None, 0, 0, geomId)
        if result != -1:
            rgba = self.environment.model.geom(geomId).rgba
        else:
            rgba = [0, 0, 0, 0]
        return 0, rgba, False, {}

def turn_done(mujoco_gym, agent):
    _healthy_z_range = (0.35, 1.3)
    if mujoco_gym.data.body(agent).xipos[2] < _healthy_z_range[0] or mujoco_gym.data.body(agent).xipos[2] > _healthy_z_range[1]:
        return True
    else:
        return False

    
def turn_reward(mujoco_gym, agent):
    _healthy_z_range = (0.35, 1.3)
    if mujoco_gym.data.body(agent).xipos[2] < _healthy_z_range[0] or mujoco_gym.data.body(agent).xipos[2] > _healthy_z_range[1]:
        return -0.5
    else:
        return 0
    
def target_reward(mujoco_gym, agent):
    if agent == "receiver":
        choices = ["choice_1", "choice_2"]
        if not "target" in mujoco_gym.data_store.keys():
                color = mujoco_gym.get_data("reference_geom")["color"]
                for choice in choices:
                    if (color == mujoco_gym.get_data(choice + "_geom")["color"]).all():
                        mujoco_gym.data_store["target"] = choice
                        mujoco_gym.data_store["target_color"] = mujoco_gym.get_data(choice + "_geom")["color"]
        
        target = mujoco_gym.data_store["target"]
        # for ankle in ["left_leg_geom_2", "left_ankle_geom_2", "right_leg_geom_2", "right_ankle_geom_2", "back_leg_geom_2", "third_ankle_geom_2", "rightback_leg_geom_2", "fourth_ankle_geom_2"]:
        for ankle in ["receiver_geom"]:
            if mujoco_gym.collision(ankle, target + "_geom"):
                return 1
            elif mujoco_gym.collision(ankle, [choice for choice in choices if choice != target][0] + "_geom"):
                return -1
    return 0

def collision_reward(mujoco_gym, agent):
    for border in ["border1_geom", "border2_geom", "border3_geom", "border4_geom", "border5_geom"]:
        # for ankle in ["left_leg_geom_2", "left_ankle_geom_2", "right_leg_geom_2", "right_ankle_geom_2", "back_leg_geom_2", "third_ankle_geom_2", "rightback_leg_geom_2", "fourth_ankle_geom_2"]:
        for ankle in [agent + "_geom"]:
            if mujoco_gym.collision(border, ankle):
                return -0.1
    return 0

def calculate_exploration_reward(mujoco_gym, agent):
    # Get the current position of the agent
    current_position = mujoco_gym.get_data(agent)["position"]
    x_t, y_t, z_t = current_position

    if "last_position" not in mujoco_gym.data_store[agent].keys():
        mujoco_gym.data_store[agent]["last_position"] = []
        mujoco_gym.data_store[agent]["last_position"].append(current_position)

    # Get the previous positions and calculate the exploration reward
    last_positions = mujoco_gym.data_store[agent]["last_position"]
    discount_factor = 0.8  # You can adjust this discount factor

    # Initialize the exploration reward
    exploration_reward = 0.0

    # Consider up to 20 previous timesteps
    num_previous_timesteps = min(len(last_positions), 10)

    for i in range(1, num_previous_timesteps + 1):
        # Calculate the Euclidean distance between current and past positions
        x_t_i, y_t_i, z_t_i = last_positions[-i]
        distance = math.sqrt((x_t - x_t_i)**2 + (y_t - y_t_i)**2)

        # Calculate the reward for this timestep and add to the exploration reward
        timestep_reward = distance * (discount_factor ** i)
        exploration_reward += timestep_reward

    # Update the last position
    mujoco_gym.data_store[agent]["last_position"].append(current_position)
    return exploration_reward

        
def target_done(mujoco_gym, agent):
    for choice in ["choice_1", "choice_2"]:
        # for ankle in ["left_leg_geom_2", "left_ankle_geom_2", "right_leg_geom_2", "right_ankle_geom_2", "back_leg_geom_2", "third_ankle_geom_2", "rightback_leg_geom_2", "fourth_ankle_geom_2"]:
        for ankle in ["receiver_geom"]:     
            if(mujoco_gym.collision(choice + "_geom", ankle)):
                return True
    return False

def border_done(mujoco_gym, agent):
    for border in ["border1_geom", "border2_geom", "border3_geom", "border4_geom", "border5_geom"]:
        if mujoco_gym.collision(border, agent + "_geom"):
            return True
    return False