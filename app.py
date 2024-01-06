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
    
    def get_observation(self):
        return self.agent_position  # Return the agent's position as the observation
    
    def get_actions(self):
        return self.action_space  # Return the action space
    
    def is_done(self):
        return self.steps_left == 0

    def action(self, action):
        if self.is_done():
            raise Exception("Game Over")
        
        self.steps_left -= 1

        # return a reward based on the action taken
        match action:
            case 0:
                self.agent_position[0] -= 1
                return -0.5
            case 1:
                self.agent_position[0] += 1
                return 0.5
            case 2:
                self.agent_position[1] += 1
                return 1.0
            case 3:
                self.agent_position[1] -= 1
                return -1.0
            case _:
                raise Exception("Invalid action")

class Agent:
    """The agent that will interact with the environment. The agent will
        randomly choose an action from the action space and receive a reward
        based on the action taken
    """

    def __init__(self):
        self.total_reward = 0.0

    def step(self, env):
        current_obs = env.get_observation()
        logger.info("Observation: %s" % current_obs)

        actions = env.get_actions()

        # Randomly choose an action
        # modify the policy to choose the action based on the observation
        policy = random.choice(actions)
        reward = env.action(policy)
        logger.info("Action: %s; Reward: %.4f" % (policy, reward))

        self.total_reward += reward

if __name__ == "__main__":
    """The main function that will run the environment and agent
    """
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)
        
    logger.info("Total reward got: %.4f" % agent.total_reward)