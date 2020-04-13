import numpy as np
import random


def sampling_pairs(num_pair, n=12):
    possibilities = set(range(1, n*(n-1)))
    keys = []
    locks = []
    for k in range(num_pair):
        key = random.sample(possibilities, 1)[0]
        key_x, key_y = key//(n-1), key%(n-1)
        lock_x, lock_y = key_x, key_y + 1
        to_remove = [key_x * (n-1) + key_y] +\
                    [key_x * (n-1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] +\
                    [key_x * (n-1) - i + key_y for i in range(1, min(2, key_y) + 1)]

        possibilities -= set(to_remove)
        keys.append([key_x, key_y])
        locks.append([lock_x, lock_y])
    agent_pos = random.sample(possibilities, 1)
    possibilities -= set(agent_pos)
    first_key = random.sample(possibilities, 1)

    agent_pos = np.array([agent_pos[0]//(n-1), agent_pos[0]%(n-1)])
    first_key = first_key[0]//(n-1), first_key[0]%(n-1)
    return keys, locks, first_key, agent_pos


colors = {0: [230, 190, 255],
          1: [170, 255, 195],
          2: [255, 250, 200],
          3: [255, 216, 177],
          4: [250, 190, 190],
          5: [240, 50, 230],
          6: [145, 30, 180],
          7: [67, 99, 216],
          8: [66, 212, 244],
          9: [60, 180, 75],
          10: [191, 239, 69],
          11: [255, 255, 25],
          12: [245, 130, 49],
          13: [230, 25, 75],
          14: [128, 0, 0],
          15: [154, 99, 36],
          16: [128, 128, 0],
          17: [70, 153, 144],
          18: [0, 0, 117]}

num_colors = len(colors)
agent_color = [128, 128, 128]
goal_color = [255, 255, 255]
grid_color = [220, 220, 220]


def world_gen(n=12, goal_length=3, num_distractor=2, distractor_length=2, seed=None, silence=False):
    """generate boxworld
    """
    if seed is not None:
        random.seed(seed)

    world_dic = {}
    world = np.ones((n, n, 3)) * 220
    goal_colors = random.sample(range(num_colors), goal_length - 1)
    distractor_possible_colors = [color for color in range(num_colors) if color not in goal_colors]
    distractor_colors = [random.sample(distractor_possible_colors, distractor_length) for k in range(num_distractor)]
    distractor_roots = random.choices(range(goal_length - 1), k=num_distractor)
    keys, locks, first_key, agent_pos = sampling_pairs(goal_length - 1 + distractor_length * num_distractor, n)


    # first, create the goal path
    for i in range(1, goal_length):
        if i == goal_length - 1:
            color = goal_color  # final key is white
        else:
            color = colors[goal_colors[i]]
        if not silence:
            print("place a key with color {} on position {}".format(color, keys[i-1]))
            print("place a lock with color {} on {})".format(colors[goal_colors[i-1]], locks[i-1]))
        world[keys[i-1][0], keys[i-1][1]] = np.array(color)
        world[locks[i-1][0], locks[i-1][1]] = np.array(colors[goal_colors[i-1]])

    # keys[0] is an orphand key so skip it
    world[first_key[0], first_key[1]] = np.array(colors[goal_colors[0]])
    if not silence:
        print("place the first key with color {} on position {}".format(goal_colors[0], first_key))

    # place distractors
    for i, (distractor_color, root) in enumerate(zip(distractor_colors, distractor_roots)):
        key_distractor = keys[goal_length-1 + i*distractor_length: goal_length-1 + (i+1)*distractor_length]
        color_lock = colors[goal_colors[root]]
        color_key = colors[distractor_color[0]]
        world[key_distractor[0][0], key_distractor[0][1] + 1] = np.array(color_lock)
        world[key_distractor[0][0], key_distractor[0][1]] = np.array(color_key)
        for k, key in enumerate(key_distractor[1:]):
            color_lock = colors[distractor_color[k-1]]
            color_key = colors[distractor_color[k]]
            world[key[0], key[1]] = np.array(color_key)
            world[key[0], key[1]+1] = np.array(color_lock)

    # place an agent
    world[agent_pos[0], agent_pos[1]] = np.array(agent_color)
    return world, agent_pos


def update_color(world, previous_agent_loc, new_agent_loc):
    world[previous_agent_loc[0], previous_agent_loc[1]] = grid_color
    world[new_agent_loc[0], new_agent_loc[1]] = agent_color


def is_empty(room):
    return np.array_equal(room, grid_color) or np.array_equal(room, agent_color)
