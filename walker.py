import sys
import enum 
from typing import Callable
import itertools

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import simpy

plt.style.use("dark_background")
STEPPER = 100


class Node:
    def __init__(self, value, parent, depth):
        self.value = value
        self.parent = parent
        self.depth = depth
        self.children = []

    def __repr__(self):
        return f'Node({self.value})'

    def add_child(self, value):        
        child = Node(value, self, self.depth+1)   
        self.children.append(child)
        return child


class Tree:
    def __init__(self, value):
        self.root = Node(value, None, 0)

    def get_all_nodes(self):
        stack = [self.root]
        nodes = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            if node.children:
                stack.extend(node.children)

        return nodes

    def find_all_leaves(self):
        stack = [self.root]
        leaves = []
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves


def scatter_tree(tree):
    values = [n.value for n in tree.get_all_nodes()]
    _ = plt.scatter([v[0] for v in values], [v[1] for v in values], 
        s=1, c='r')
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.show()
    

def graph_tree(tree):
    plt.figure(figsize=(10,10))
    stack = [tree.root]
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


class WalkerBase:
    '''
    The walker objects must implement a decide_action that is equivalent
    to what could be conceived as Policies for RL. The walker object
    will encounter some environment that will be determined stochastically
    and will decide what to do based on what the user defined which itself
    is basing his decision in terms what could be interpreted as a state.
    '''
    ACTIONS = enum.Enum('ACTIONS', 'MOVE STAY EXTEND')
    PARENT_SPLIT_PROBABILITY = 0.0

    def __init__(self, env: simpy.Environment, decide: Callable=None, d: int=2, 
        m0: int=10, E0: float=10):
        self.env = env
        self.d = d
        self.mass = m0
        self.E = E0
        self.body = Tree(np.zeros(d, dtype=float))
        self.N = 1
        self.history = []
        self.nodes = [self.body.root]
        self.decide = decide 

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
            child = leaf_to_expand.add_child(new_leaf_value)
            self.nodes.append(child)

    def move(self):
        displacement = np.random.randint(-1, 2, self.d) 
        for node in self.nodes:
            node.value += displacement

    def update(self, save=True) -> ACTIONS: 
        action = self.ACTIONS.EXTEND if not self.decide else None
        if not action:
            action = self.decide(self.body, self.N, self.E) 
        
        if save:
            self.history.append([n.value.copy() 
                for n in self.body.get_all_nodes()])

        if action == self.ACTIONS.EXTEND:
            self.extend()
        if action == self.ACTIONS.MOVE:
            self.move()
        elif action == self.ACTIONS.STAY:
            pass

    def draw(self, method):
        if method == 'scatter':
            scatter_tree(self.body)
        elif method == 'graph':
            graph_tree(self.body)

    def run(self, save=False):
        while True: 
            yield self.env.timeout(1)
            self.update(save)
        

def possible_displacements(d):
    elements = [-1, 0, 1]
    return list(itertools.product(elements, repeat=d))


class DepthWalker(WalkerBase):
    def extend(self, save=True):
        try:
            self.nodes_depth
        except AttributeError:
            self.nodes_depth = {n.depth:[] for n in self.nodes}
            for n in self.nodes:
                self.nodes_depth[n.depth].append(n)

        try:
            self.displacements
        except:
            self.displacements = np.array(possible_displacements(self.d))
    
        depths = list(self.nodes_depth.keys())
        p = np.array([np.sqrt(d+1) for d in depths])
        center_of_mass = np.array([n.value for n in self.nodes]).mean(axis=0)
        max_depth = max(depths)
        if np.random.random() < 0.99:
            p = p/p.sum()
            depth = np.random.choice(depths, p=p)
        else:
            depth = max_depth
        node_to_expand = np.random.choice(self.nodes_depth[depth])
        
        dist = 1
        delta = np.random.randint(-dist, dist+1, self.d)
        if len(self.nodes) != 1:
            if np.random.random() > 0.3:
                delta = (node_to_expand.value-center_of_mass)
                delta = delta/np.linalg.norm(delta)
                diffs = np.linalg.norm(self.displacements-delta, axis=1)
                closest3 = np.argsort(diffs)[:5]
                delta = self.displacements[np.random.choice(closest3)]

        new_leaf_value = node_to_expand.value + delta
        if not any([all(l.value == new_leaf_value) for l in self.nodes]):
            self.N +=1
            child = node_to_expand.add_child(new_leaf_value)
            self.nodes.append(child)
            if not self.nodes_depth.get(child.depth):
                self.nodes_depth[child.depth] = [child]
            else:
                self.nodes_depth[child.depth].append(child)


def random_decider(body, N, E):
    return np.random.choice(list(WalkerBase.ACTIONS))


def extend(_, __, ___):
    return WalkerBase.ACTIONS.EXTEND


def walkerbase_example():
    env = simpy.Environment()
    walker = WalkerBase(env, random_decider, 2)
    env.process(walker.run())
    for i in tqdm.trange(1, int(sys.argv[1])//STEPPER):
        env.run(until=i*STEPPER)

    walker.draw('graph')    
    walker.draw('scatter')


def depthwalker_example():
    env = simpy.Environment()
    walker = DepthWalker(env, random_decider, 2)
    env.process(walker.run())
    for i in tqdm.trange(1, int(sys.argv[1])//STEPPER):
        env.run(until=i*STEPPER)

    
    walker.draw('graph')    
    walker.draw('scatter') 
    


if __name__ == "__main__":
    #walkerbase_example()    
    depthwalker_example()