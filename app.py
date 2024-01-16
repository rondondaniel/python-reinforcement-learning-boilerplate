import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

class Environment:
    """A simple environment with a fixed set of states and actions. The agent
    can move in four directions, and the goal is to reach the goal state in
    as few steps as possible. The agent receives a reward based on the action
    """

    def __init__(self):
        self.steps_left = 10
        self.agent_position = [0, 0]  # New attribute to store the agent's position
        self.goal_position = [4, 4]  # New attribute to store the goal's position
        self.action_space = [0, 1, 2, 3]  # New attribute to store the action space
 
    def reset(self):
        self.steps_left = 10
        self.agent_position = [0, 0]
        
        return self._get_observation()
    
    def get_actions(self):
        return self.action_space
    
    def _get_goal_position(self):
        return self.goal_position
    
    def _get_observation(self):
        return {
            "distance_to_goal": abs(self.agent_position[0] - self.goal_position[0]) + abs(self.agent_position[1] - self.goal_position[1]),
            "agent_position": self.agent_position 
        }
    
    def _is_done(self):
        return self.steps_left == 0
    
    def _move(self, action):
        if action == 0:
            self.agent_position[0] -= 1
        elif action == 1:
            self.agent_position[0] += 1
        elif action == 2:
            self.agent_position[1] += 1
        elif action == 3:
            self.agent_position[1] -= 1
        else:
            raise Exception("Invalid action")
        
        self.steps_left -= 1
    
    def _get_reward(self, observation):
        # Get the distance to the goal and position from the observation
        distance_to_goal = observation["distance_to_goal"]
        agent_position = observation["agent_position"]

        # If the agent is at the goal position, return a large positive reward
        if agent_position == self._get_goal_position():
            return 100.0
        # If the agent is not at the goal position, return a penality proportional to the distance to the goal
        else:
            return -distance_to_goal

    def step(self, action):
        info = {}
        reward = 0.0

        # Move the agent
        self._move(action)

        # Get the new state, reward, done and info
        observations = self._get_observation()
        reward = self._get_reward(observations)
        done = self._is_done()
        
        # return the next state, reward, done and info
        return observations, reward, done, info

class Agent:
    """The agent that will interact with the environment. The agent will
        randomly choose an action from the action space and receive a reward
        based on the action taken
    """

    def __init__(self, env):
        self._total_reward = 0.0
        self._action_space = env.get_actions()  # New attribute to store the action space
        self._last_action = None
        self._last_reward = None

    @property
    def action_space(self):
        return self._action_space  # Return the action space

    @property
    def last_action(self):
        return self._last_action

    @last_action.setter
    def last_action(self, action):
        self._last_action = action

    @property
    def total_reward(self):
        return self._total_reward

    @total_reward.setter
    def total_reward(self, reward):
        self._total_reward += reward

    def choose_action(self, observation):
        # Get the list of possible actions
        actions = self.action_space

        # Randomly choose an action
        # modify the policy to choose the action based on the observation
        # use the observation to choose the action based on the distance to the goal
        if observation["distance_to_goal"] > 0:
            action = random.choice(actions)
        else:
            action = None

        # Set the last action
        self.last_action = action
        
        return action
        
if __name__ == "__main__":
    """The main function that will run the environment and agent
    """
    
    done = False
    env = Environment()
    agent = Agent(env)

    # Reset the environment and agent
    current_obs = env.reset()

    # Loop until the environment is done
    while not done:
        # Log the current observation
        logger.info("Observation: %s" % current_obs)

        # Choose an action based on the current observation
        action = agent.choose_action(current_obs)

        # Take the action and get the reward
        new_state, reward, done, info = env.step(action)
        logger.info("Action: %s; Reward: %.4f" % (action, reward))
        logger.info("Next State: %s" % new_state)
        logger.info("---------------------------------------------------------------")

        # Update the total reward
        agent.total_reward = reward
        # Update the current observation
        current_obs = new_state

    logger.info("Episode done!")
    logger.info("Total reward got: %.4f" % agent.total_reward)