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
 
    def reset(self):
        self.steps_left = 10
        self.agent_position = [0, 0]
    
    def get_observation(self):
        return self.agent_position  # Return the agent's position as the observation
    
    
    def is_done(self):
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
    
    def _get_reward(self):
        if self.agent_position == self.goal_position:
            return 1.0
        else:
            return 0.0

    def step(self, action):
        info = {}
        reward = 0.0

        # Move the agent
        self._move(action)

        # get the reward and check if the game is over
        reward = self._get_reward()
        next_state = self.agent_position
        done = self.is_done()
            
        # return the next state, reward, done and info
        return next_state, reward, done, info

class Agent:
    """The agent that will interact with the environment. The agent will
        randomly choose an action from the action space and receive a reward
        based on the action taken
    """

    def __init__(self):
        self.total_reward = 0.0
        self.action_space = [0, 1, 2, 3]  # New attribute to store the action space

    def _get_actions(self):
        return self.action_space  # Return the action space
    
    def choose_action(self, observation):
        # Get the list of possible actions
        actions = self._get_actions()

        # Randomly choose an action
        # modify the policy to choose the action based on the observation
        action = random.choice(actions)
        
        return action
        
if __name__ == "__main__":
    """The main function that will run the environment and agent
    """
    
    done = False
    env = Environment()
    agent = Agent()

    # Reset the environment
    env.reset()
    current_obs = env.get_observation()

    # Loop until the environment is done
    while not done:
        # Log the current observation
        logger.info("Observation: %s" % current_obs)

        # Choose an action based on the current observation
        action = agent.choose_action(current_obs)

        # Take the action and get the reward
        next_state, reward, done, info = env.step(action)
        logger.info("Action: %s; Reward: %.4f" % (action, reward))
        logger.info("Next State: %s" % next_state)

        # Update the total reward
        agent.total_reward += reward
        # Update the current observation
        current_obs = next_state

    logger.info("Episode done!")
    logger.info("Total reward got: %.4f" % agent.total_reward)