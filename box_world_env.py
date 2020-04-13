import time
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from boxworld_gen import *


class boxworld(gym.Env):
    """Boxworld representation
    Args:
      n: specify the size of the field (n x n)
      goal_length
      num_distractor
      distractor_length
      world: an existing world data. If this is given, use this data.
             If None, generate a new data by calling world_gen() function
    """

    def __init__(self, n, goal_length, num_distractor, distractor_length,
                 viewport_size=5, max_steps=300, world=None, silence=False):
        self.goal_length = goal_length
        self.num_distractor = num_distractor
        self.distractor_length = distractor_length
        self.viewport_size = viewport_size
        self.n = n
        self.num_pairs = goal_length - 1 + distractor_length * num_distractor

        # Penalties and Rewards
        self.step_cost = 0.1
        self.reward_gem = 10
        self.reward_key = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.observation_space = Box(low=0, high=255, shape=(n, n, 3), dtype=np.uint8)
        self.silence = silence

        # Game initialization
        self.owned_key = np.array(grid_color, dtype=np.float64)

        self.np_random_seed = None

        self.world = None
        self.reset(world)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.img = self.ax.imshow(self.world, vmin=0, vmax=255, interpolation='none')
        self.fig.canvas.draw()
        self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show(block=False)

    def seed(self, seed=None):
        self.np_random_seed = seed
        return [seed]

    def save(self):
        np.save('box_world.npy', self.world)

    def step(self, action):

        change = CHANGE_COORDINATES[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        self.num_env_steps += 1

        reward = -self.step_cost
        done = self.num_env_steps == self.max_steps

        # Move player if the field in the moving direction is either

        if np.any(new_position < 0) or np.any(new_position >= self.n):
            possible_move = False

        elif np.array_equal(new_position, [0, 0]):
            possible_move = False

        elif is_empty(self.world[new_position[0], new_position[1]]):
            # No key, no lock
            possible_move = True

        elif new_position[1] == 0 or is_empty(self.world[new_position[0], new_position[1]-1]):
            # It is a key
            if is_empty(self.world[new_position[0], new_position[1]+1]):
                # Key is not locked
                possible_move = True
                self.owned_key = self.world[new_position[0], new_position[1]].copy()
                # self.world[0, 0] = self.owned_key
                if np.array_equal(self.world[new_position[0], new_position[1]], goal_color):
                    # Goal reached
                    reward += self.reward_gem
                    done = True
                else:
                    reward += self.reward_key
            else:
                possible_move = False
        else:
            # It is a lock
            if np.array_equal(self.world[new_position[0], new_position[1]], self.owned_key):
                # The lock matches the key
                self.owned_key = np.array(grid_color, dtype=np.float64)
                possible_move = True
            else:
                possible_move = False
                if not self.silence:
                    print("lock color is {}, but owned key is {}".format(
                        self.world[new_position[0], new_position[1]], self.owned_key))

        if possible_move:
            self.player_position = new_position
            update_color(self.world, previous_agent_loc=current_position, new_agent_loc=new_position)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": possible_move,
        }

        return self.state(), reward, done, info

    def reset(self, world=None):
        if world is None:
           self.world, self.player_position = world_gen(n=self.n, goal_length=self.goal_length,
                                                        num_distractor=self.num_distractor,
                                                        distractor_length=self.distractor_length,
                                                        seed=self.np_random_seed,
                                                        silence=self.silence)
        else:
            self.world, self.player_position = world

        self.num_env_steps = 0

        return self.state()

    def render(self, mode='window'):
        img = self.world_fog_map()
        if mode == 'return':
            return img
        else:
            self.img.set_data(img)
            self.fig.canvas.restore_region(self.axbackground)
            self.ax.draw_artist(self.img)
            self.fig.canvas.blit(self.ax.bbox)
            plt.pause(0.001)

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def viewport_mask(self):
        mask = np.zeros(shape=(self.n, self.n), dtype=np.bool)
        k = self.viewport_size // 2
        t = max(self.player_position[0] - k, 0)
        b = min(self.player_position[0] + k + 1, self.n)
        l = max(self.player_position[1] - k, 0)
        r = min(self.player_position[1] + k + 1, self.n)
        mask[t:b, l:r] = True
        return mask

    def viewport_map(self):
        k = self.viewport_size // 2
        world = np.pad(self.world, ((k, k), (k, k), (0, 0)),
                       mode='constant', constant_values=0)
        t = self.player_position[0] + k - k
        b = self.player_position[0] + k + k + 1
        l = self.player_position[1] + k - k
        r = self.player_position[1] + k + k + 1
        return world[t:b, l:r, :]

    def world_fog_map(self):
        img = self.world.copy()
        mask_out_of_viewport = np.logical_not(self.viewport_mask())
        mask_grid_color = np.ma.masked_equal(self.world, grid_color).mask[:, :, 0]
        mask = np.logical_and(mask_out_of_viewport, mask_grid_color)
        img[mask] = 0
        img = img.astype(np.uint8)
        return img

    def state(self):
        return self.viewport_map() / 255.0,\
               self.owned_key / 255.0,\
               self.player_position / self.n


ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


if __name__ == "__main__":
    # execute only if run as a script
    env = boxworld(12, 3, 2, 1)
    # env.seed(1)
    env.reset()
    env.render()
