import gymnasium as gym
from gymnasium import spaces
import numpy as np
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | <yellow>{level}</yellow> | <blue>{message}</blue>,")

class GridWorldEnv(gym.Env):
    """
    Standard Gymnasium environment for a 2D Grid World.
    Goal: Navigate to the bottom-right corner in the fewest steps.
    """
    def __init__(self, grid_size: int = 5, max_steps: int = 50):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Action space: 0: Up, 1: Down, 2: Right, 3: Left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: (x, y) coordinates bounding box
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([grid_size - 1, grid_size - 1]), 
            dtype=np.int32
        )
        
        self._agent_location = np.array([0, 0], dtype=np.int32)
        self._target_location = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)
        self._steps_taken = 0

    def _get_obs(self) -> np.ndarray:
        # Return a copy to prevent the agent from mutating internal state
        return self._agent_location.copy()

    def _get_info(self) -> dict:
        # Provide auxiliary information (e.g., Manhattan distance)
        distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        return {"distance_to_goal": distance}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps_taken = 0
        self._agent_location = np.array([0, 0], dtype=np.int32)
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Map discrete action to 2D vector direction
        direction = np.array([0, 0])
        if action == 0:   direction = np.array([-1, 0])  # Up
        elif action == 1: direction = np.array([1, 0])   # Down
        elif action == 2: direction = np.array([0, 1])   # Right
        elif action == 3: direction = np.array([0, -1])  # Left

        # Update location and explicitly clip to grid boundaries using NumPy
        self._agent_location = np.clip(
            self._agent_location + direction, 
            0, 
            self.grid_size - 1
        )
        
        self._steps_taken += 1

        # Check termination (reached the goal) and truncation (hit time limit)
        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = bool(self._steps_taken >= self.max_steps)

        # Sparse reward mechanism: +100 for goal, -1 step penalty to encourage speed
        reward = 100.0 if terminated else -1.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()


class QLearningAgent:
    """
    A foundational tabular Q-Learning agent using an epsilon-greedy policy.
    """
    def __init__(self, action_space: spaces.Discrete, learning_rate: float = 0.1, discount_factor: float = 0.99, epsilon: float = 0.1):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Dictionary allows handling of arbitrary discrete coordinate states without pre-allocating a 2D matrix
        self.q_table = {} 

    def _get_q_values(self, state: tuple) -> np.ndarray:
        # Initialize state with zeros if unseen
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        return self.q_table[state]

    def choose_action(self, obs: np.ndarray) -> int:
        state = tuple(obs)
        # Epsilon-greedy exploration strategy
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore: random action
        else:
            return int(np.argmax(self._get_q_values(state)))  # Exploit: best known action

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool):
        state = tuple(obs)
        next_state = tuple(next_obs)
        
        q_values = self._get_q_values(state)
        next_q_values = self._get_q_values(next_state)
        
        # Bellman equation application
        target = reward
        if not terminated:
            target += self.gamma * np.max(next_q_values)
            
        q_values[action] += self.lr * (target - q_values[action])


if __name__ == "__main__":
    # Initialize the standardized environment and agent
    env = GridWorldEnv(grid_size=15, max_steps=150)
    agent = QLearningAgent(action_space=env.action_space, learning_rate=0.1, epsilon=0.1)

    episodes = 500
    
    # Training Loop
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # 1. Select action
            action = agent.choose_action(obs)
            
            # 2. Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Update agent policy (Learning step)
            agent.update(obs, action, reward, next_obs, terminated)
            
            # 4. Transition state
            obs = next_obs
            total_reward += reward
            done = terminated or truncated

        # Logging formatting modernized to f-strings
        if (episode + 1) % 100 == 0:
            logger.info(f"Episode: {episode + 1}/{episodes} | Total Reward: {total_reward:.2f} | Final Distance: {info['distance_to_goal']}")

    logger.info("Training complete.")
