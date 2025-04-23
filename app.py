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
        self._steps_left = None
        self._agent_position = None
        self._goal_position = [4, 4]  # New attribute to store the goal's position
        self._action_space = [0, 1, 2, 3]  # New attribute to store the action space

    @property
    def steps_left(self):
        return self._steps_left

    @steps_left.setter
    def steps_left(self, steps):
        self._steps_left = steps

    @property
    def agent_position(self):
        return self._agent_position

    @agent_position.setter
    def agent_position(self, position):
        self._agent_position = position

    @property
    def goal_position(self):
        return self._goal_position

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self.steps_left = 100
        self.agent_position = [0, 0]
        
        return self._get_observation()

    def _get_observation(self):
        return {
            "distance_to_goal": abs(self.agent_position[0] - self.goal_position[0]) + abs(self.agent_position[1] - self.goal_position[1]),
            "agent_position": self.agent_position 
        }

    def _is_done(self):
        return self.steps_left == 0
    
    def _move(self, action):
        # Move the agent based on the action chosen
        match action:
            case 0:
                self.agent_position[0] -= 1
            case 1:
                self.agent_position[0] += 1
            case 2:
                self.agent_position[1] += 1
            case 3:
                self.agent_position[1] -= 1
            case _:
                raise Exception("Invalid action")
            
        # avoid the agent to go out of the grid
        if self.agent_position[0] < 0:
            self.agent_position[0] = 0
        if self.agent_position[0] > 4:
            self.agent_position[0] = 4
        if self.agent_position[1] < 0:
            self.agent_position[1] = 0
        if self.agent_position[1] > 4:
            self.agent_position[1] = 4
        
        # Decrease the number of steps left
        self.steps_left -= 1
    
    def _get_reward(self, observation):
        # Get the distance to the goal and position from the observation
        distance_to_goal = observation["distance_to_goal"]
        agent_position = observation["agent_position"]

        # If the agent is at the goal position, return a large positive reward
        if agent_position == self.goal_position:
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
        self._action_space = env.action_space  # New attribute to store the action space
        self._last_action = None
        self._last_total_reward = 0.0
        self._last_distance_to_goal = 8.0

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
    
    @property
    def last_total_reward(self):
        return self._last_total_reward
    
    @last_total_reward.setter
    def last_total_reward(self,total_reward):
        self._last_total_reward = total_reward

    @property
    def last_distance_to_goal(self):
        return self._last_distance_to_goal
    
    @last_distance_to_goal.setter
    def last_distance_to_goal(self, distance_to_goal):
        self._last_distance_to_goal = distance_to_goal

    def choose_action(self, observation):
        # Get the list of possible actions
        actions = self.action_space

        # Randomly choose an action
        # modify the policy to choose the action based on the observation
        # use the observation to choose the action based on the distance to the goal
        if self.total_reward <= self.last_total_reward and self.last_distance_to_goal <= observation["distance_to_goal"]:
            action = random.choice(actions)
        else:
            if self.last_action is None :
                action = random.choice(actions)
            else:
                action = self.last_action

        # Set the last distance to goal and last action
        self.last_distance_to_goal = observation["distance_to_goal"]
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
        logger.info("Steps left: %d" % env.steps_left)
        logger.info("Observation: %s" % current_obs)
        logger.info("Total reward: %.4f" % agent.total_reward)

        # Choose an action based on the current observation
        action = agent.choose_action(current_obs)

        # Take the action and get the reward
        new_state, reward, done, info = env.step(action)
        logger.info("Action: %s; Reward: %.4f" % (action, reward))
        logger.info("Next State: %s" % new_state)
        logger.info("---------------------------------------------------------------")

        # Update the total reward
        agent.last_total_reward = agent.total_reward
        agent.total_reward = reward
        # Update the current observation
        current_obs = new_state

    logger.info("Episode done!")
    logger.info("Total reward got: %.4f" % agent.total_reward)