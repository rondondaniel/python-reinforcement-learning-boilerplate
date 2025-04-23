import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.action_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

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
        # PPO policy network
        obs_dim = 2  # distance_to_goal, agent_position (flattened)
        act_dim = len(self._action_space)
        self.policy_net = PolicyNetwork(obs_dim, act_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.memory = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "logprobs": [],
            "values": []
        }

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

    def store_transition(self, obs, action, reward, done, logprob, value):
        self.memory["obs"].append(obs)
        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)
        self.memory["logprobs"].append(logprob)
        self.memory["values"].append(value)

    def _preprocess_observation(self, observation):
        # Flatten the observation dict into a tensor
        distance = observation["distance_to_goal"]
        pos = observation["agent_position"]
        return torch.tensor([distance, pos[0] + pos[1]], dtype=torch.float32, device=self.device)
    
    def take_action(self, observation):
        # PPO policy: use the policy network to select an action
        obs_tensor = self._preprocess_obs(observation)
        logits = self.policy_net(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        logprob = dist.log_prob(action)
        value = self.policy_net(obs_tensor)[1]
        # Set the last distance to goal and last action
        self.last_distance_to_goal = observation["distance_to_goal"]
        self.last_action = action
        return action, logprob.item(), value.item(), obs_tensor.detach().cpu().numpy()
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        rewards = self.memory["rewards"]
        dones = self.memory["dones"]
        values = self.memory["values"] + [last_value]
        gae = 0
        returns = []
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advantages

    def ppo_update(self, epochs=4, batch_size=32, clip_epsilon=0.2, gamma=0.99, lam=0.95, lr=3e-4):
        obs = torch.tensor(self.memory["obs"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.memory["logprobs"], dtype=torch.float32, device=self.device)
        
        # Bootstrap value for last state
        with torch.no_grad():
            last_value = self.policy_net(obs[-1])[1].item()
            
        returns, advantages = self.compute_returns_and_advantages(last_value, gamma, lam)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        n = len(obs)
        for _ in range(epochs):
            idxs = torch.randperm(n)
            for start in range(0, n, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                logits, values_pred = self.policy_net(mb_obs)
                probs = F.softmax(logits, dim=-1)
                dist_cat = torch.distributions.Categorical(probs)
                new_logprobs = dist_cat.log_prob(mb_actions)
                entropy = dist_cat.entropy().mean()
                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, mb_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def clear_memory(self):
        for k in self.memory:
            self.memory[k] = []
        
if __name__ == "__main__":
    """The main function that will run the environment and agent
    """
    
    done = False
    env = Environment()
    agent = Agent(env)

    # Reset the environment and agent
    current_obs = env.reset()
    steps_count = 0
    PPO_UPDATE_STEPS = 2048

    # Loop until the environment is done
    while not done:
        # Log the current observation
        logger.info("Steps left: %d" % env.steps_left)
        logger.info("Observation: %s" % current_obs)
        logger.info("Total reward: %.4f" % agent.total_reward)

        # Choose an action based on the current observation
        action, logprob, value, obs = agent.take_action(current_obs)

        # Take the action and get the reward
        new_state, reward, done, info = env.step(action)
        logger.info("Action: %s; Reward: %.4f" % (action, reward))
        logger.info("Next State: %s" % new_state)
        logger.info("---------------------------------------------------------------")

        # Store transition
        agent.store_transition(obs, action, logprob, reward, done, value)
        step_count += 1
        
        # Update the total reward
        agent.last_total_reward = agent.total_reward
        agent.total_reward = reward
        # Update the current observation
        current_obs = new_state

        # PPO update every PPO_UPDATE_STEPS
        if step_count % PPO_UPDATE_STEPS == 0 or done:
            agent.ppo_update()
            agent.clear_memory()

    logger.info("Episode done!")
    logger.info("Total reward got: %.4f" % agent.total_reward)