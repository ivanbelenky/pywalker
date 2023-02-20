import sys
import enum 
from typing import Callable

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import simpy

plt.style.use("dark_background")


STEPPER = 100


class Tree:
    def __init__(self, value, parent=None, children=None):
        if parent is not None and not isinstance(parent, Tree):
            raise TypeError('Parent must be a Tree instance')
        self.value = value
        self.parent = parent
        self.children = children or []
        self.is_root = parent is None

    def add_child(self, child):
        if not isinstance(child, Tree):
            raise TypeError('Child must be a Tree instance')
        self.children.append(child)
        child.parent = self
    
    def get_all_nodes(self):
        stack = [self]
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            if node.children:
                stack.extend(node.children)

        return nodes

    def find_all_leaves(self):
        stack = [self]
        leaves = []
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves

class Walker:
    '''
    The walker objects must implement a decide_action that is equivalent
    to what could be conceived as Policies for RL. The walker object
    will encounter some environment that will be determined stochastically
    and will decide what to do based on what the user defined which itself
    is basing his decision in terms what could be interpreted as a state.
    '''
    ACTIONS = enum.Enum('ACTIONS', 'MOVE STAY EXTEND')
    PARENT_SPLIT_PROBABILITY = 0.0

    def __init__(self, env: simpy.Environment, decide: Callable, d: int=2, 
        m0: int=10, E0: float=10):
        self.env = env
        self.d = d
        self.mass = m0
        self.E = E0
        self.body = Tree(np.zeros(d, dtype=int))
        self.N = 1
        self.history = []
        self.decide = decide
        
    def decide_action(self):
        return self.decide(self.body, self.N, self.E)

    def extend(self):
        all_leaves = self.body.find_all_leaves()
        all_parents = [l.parent for l in all_leaves if l.parent]
        
        parent = np.random.random() < self.PARENT_SPLIT_PROBABILITY
        selected = all_parents if parent and all_parents else all_leaves

        leaf_to_expand = np.random.choice(selected) 
        
        dist = 1

        new_leaf_value = leaf_to_expand.value + np.random.randint(-dist, dist+1, self.d)
        if not any([all(l.value == new_leaf_value) for l in all_leaves]):
            self.N +=1
            leaf_to_expand.add_child(Tree(new_leaf_value))

    def move(self):
        all_nodes = self.body.get_all_nodes()
        displacement = np.random.randint(-1, 2, self.d) 
        for node in all_nodes:

            node.value += displacement

    def update(self, save=True) -> ACTIONS:
        action = self.decide_action()
        if save:
            self.history.append([n.value.copy() 
                for n in self.body.get_all_nodes()])

        action = self.decide_action()
        if action == self.ACTIONS.EXTEND:
            self.extend()
        if action == self.ACTIONS.MOVE:
            self.move()
        elif action == self.ACTIONS.STAY:
            pass
        
        # less energy per update right?

    def run(self, save=False):
        while True: 
            yield self.env.timeout(1)
            self.update(save)


def random_decider(body, N, E):
    return Walker.ACTIONS.EXTEND


# make a function that grabs a list of x,y int positions and plots a grid 
# with the positions as black squares

def scatter_tree(walker):
    values = [n.value for n in walker.body.get_all_nodes()]
    _ = plt.scatter([v[0] for v in values], [v[1] for v in values], 
        s=1, c='r')
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.show()
    


def  graph_tree(walker):
    plt.figure(figsize=(10,10))
    stack = [walker.body]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
            for child in node.children:
                nval, cval = node.value, child.value
                plt.plot([nval[0], cval[0]], [nval[1], cval[1]], 
                    linewidth=0.1, c='r')
    _ = plt.xticks([])
    _ = plt.yticks([])
    
    plt.show()


if __name__ == "__main__":
    env = simpy.Environment()
    walker = Walker(env, random_decider, 2)
    env.process(walker.run())
    for i in tqdm.trange(1, int(sys.argv[1])//STEPPER):
        env.run(until=i*STEPPER)

    graph_tree(walker)    
    scatter_tree(walker)