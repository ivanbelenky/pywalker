import sys
import enum 
from copy import deepcopy
import itertools
from typing import Callable

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import simpy
from celluloid import Camera

plt.style.use("dark_background")

'''
I am thinking about building an arbitrary graph, setting up distances based
on plain euclidean distance. Then I want to connect the nodes with edges
at least 1 connection per graph point. 

Then I just want to simulate transport of mass between nodes from one node 
to other that is connected. Each node has a mass, each time step, 
the node will decide if it wants to move mass or not. If it does, it will
timeout the distance divided by the velocity. 

'''


class TransfererBase:
    PROB_TO_TRANSFER = 0.3
    SPEED = 100
    MASS_TO_TRANSFER = 10

    def __init__(self, n, env, m0, size=(30, 30), r0=None):
        self.env = env
        self.m0 = m0
        self.size = size
        if r0:
            self.nodes = {i:{'r': r0[i], 'm': m0} for i in range(n)}
        else:
            self.nodes = {i:{'r': np.random.random(2)*size, 'm': m0} for i in range(n)}
        self.history = []
        self.velocity = self.SPEED
        self.edges = self.set_edges()
        self.history = []

    def set_edges(self):
        edges = {k:{} for k in self.nodes.keys()}
        node_keys = list(self.nodes.keys())
        for node in self.nodes.keys():
            n_edges = np.random.choice([n for n in self.nodes.keys() if n != node])
            tos = np.random.choice(node_keys, size=n_edges, replace=False)
            for to in tos:
                Ar = self.nodes[node]['r']
                Br = self.nodes[to]['r']
                d = np.linalg.norm(Ar-Br)
                edges[node][to] = d
                edges[to][node] = d
        return edges

    def record(self, A, B, m, duration):
        now = self.env.now
        self.history.append((now, duration, A, B, m))

    def transfer(self, A, B, m):
        timeout = self.edges[A][B] // self.velocity + 1
        self.nodes[A]['m'] -= m
        self.record(A, B, m, timeout)
        yield self.env.timeout(timeout)
        self.nodes[B]['m'] += m

    def should_transfer(self, node: dict) -> bool:
        return np.random.random() < self.PROB_TO_TRANSFER

    def mass_to_transfer(self, node: dict) -> int:
        return self.m0//10+1

    def cant_transfer(self, A, mass_to_transfer):
        return self.nodes[A]['m'] - mass_to_transfer < 1

    def pick_node_to_transfer(self, A) -> int:
        return np.random.choice(list(self.edges[A].keys()))
        
    def update(self):
        while True:
            for node in self.nodes.keys():
                if self.should_transfer(self.nodes[node]):
                    A = node 
                    mass_to_transfer = self.mass_to_transfer(self.nodes[node])
                    if self.cant_transfer(A, mass_to_transfer):
                        continue
                    B = self.pick_node_to_transfer(A)
                    self.env.process(self.transfer(A, B, mass_to_transfer))
            yield self.env.timeout(1)
            
    def get_transfer_dict(self):
        transfer_dict = {}
        for h in self.history:
            now, duration, A, B, m = h
            if now not in transfer_dict:
                transfer_dict[now] = []
            transfer_dict[now].append((duration, A, B, m))
        return transfer_dict

    def plot_history(self, to_plot, save, filepath):
        fig = plt.figure(figsize=(10,10))
        camera = Camera(fig)

        plt.xlim(0, self.size[0])
        plt.ylim(0, self.size[1])
        plt.xticks([])
        plt.yticks([])

        to_plot = list(to_plot.values())
        for i in tqdm(range(len(to_plot))):
            v = to_plot[i]
            plt.xlim(0, self.size[0])
            plt.ylim(0, self.size[1])
            plt.xticks([])
            plt.yticks([])
            
            node_masses = v['node_masses']
            travelling_pos = v['travelling']
            
            for node in node_masses:
                plt.scatter(self.nodes[node]['r'][0], self.nodes[node]['r'][1], 
                           s=node_masses[node]*10, color='red', alpha=0.5)
            #plot travelling
            #for pos in travelling_pos:
            #    plt.scatter(pos[0], pos[1], color='white', alpha=0.5)
            
            camera.snap()
        anim = camera.animate(blit=True)
        if save:
            anim.save(filepath, fps=20)

    def play_history(self, save=True, filepath='./assets/history.gif'):
        max_dt = self.history[-1][0]
        transfer_dict = self.get_transfer_dict()
        travelling = {}
        node_masses = {k:self.m0 for k in self.nodes.keys()}

        to_plot ={i:{'travelling': {}, 'node_masses': {}} for i in range(max_dt)}
        for i in range(max_dt):
            to_plot[i]['travelling'] = deepcopy([v['pos'] for v in travelling.values()])
            to_plot[i]['node_masses'] = deepcopy(node_masses)
            to_pop = []    
            for k in travelling:
                travelling[k]['eta'] -= 1
                travelling[k]['pos'] = travelling[k]['pos'] + \
                    travelling[k]['vel']

                if travelling[k]['eta'] == 0:
                    A, B, _ = k.split('-')
                    A, B = int(A), int(B)
                    node_masses[B] += travelling[k]['m']
                    to_pop.append(k)

            for k in to_pop:
                travelling.pop(k)
            
            transfers_i = transfer_dict.get(i, [])
            for transfer in transfers_i:
                duration, A, B, m = transfer
                node_masses[A] -= m
                travelling[f'{A}-{B}-{i}'] = {
                    'pos': self.nodes[A]['r'], 
                    'vel': (self.nodes[B]['r']-self.nodes[A]['r'])/duration,  
                    'm': m,
                    'eta': duration
                }

        self.plot_history(to_plot, save, filepath)


class ETransferer(TransfererBase):
    def should_transfer(self, node: dict) -> bool:
        return np.random.random() < node['m']/self.m0
    
    def mass_to_transfer(self, node: dict) -> int:
        return np.random.normal(node['m']//10 + 1, 4)

    def pick_node_to_transfer(self, A) -> int:
        distances = [self.edges[A][B] for B in self.edges[A].keys()]
        p = distances/np.sum(distances)
        return np.random.choice(list(self.edges[A].keys()), p=p) 


class FriendsTransferers(TransfererBase):
    def pick_node_to_transfer(self, A) -> int:
        connections = list(self.edges[A].keys())
        distances = [self.edges[A][B] for B in self.edges[A].keys()]
        sorted_d = np.argsort(distances)
        if np.random.random() < 0.2:
            return connections[sorted_d[0]]
        else:
            return np.random.choice(connections)

    def mass_to_transfer(self, node: dict) -> int:
        return np.random.normal(node['m']//10 + 1, 4)


if __name__ ==  "__main__":
    env = simpy.Environment()


    r0 = []
    for i in range(49):
        r0.append(np.array([i%7+1, i//7+1]))
    graph = FriendsTransferers(49, env, 100, size=(8,8), r0=r0)
    env.process(graph.update())
    for i in tqdm(range(1,2000)):
        env.run(until=i)

    graph.play_history(save=True, filepath='./assets/grid.gif')
