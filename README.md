# Walker

Something can emerge from this. At the moment just random walkish behavior. 

```python
env = simpy.Environment()
walker = Walker(env, random_decider, 3)
env.process(walker.run())
env.run(until=1000)
```


<p align="center">
  <img src="https://github.com/ivanbelenky/pywalker/blob/master/assets/graph_walker.png">
</p>

<p align="center">
  <img src="https://github.com/ivanbelenky/pywalker/blob/master/assets/mediumgraph15K.png">
</p>

<p align="center">
  <img src="https://github.com/ivanbelenky/pywalker/blob/master/assets/supergraph.png">
</p>


# Transferer

Randomly connected graph with transfer rules for mass. Its like a brownian expansion of the nodes of the graph but with underlying stochastic rules. So it does not look totally cahotic. There is some conservation.


```python
graph = FriendsTransferers(49, env, 100, size=(8,8))
env.process(graph.update())
env.run(until=2000)
graph.play_history(save=True, filepath='./assets/grid.gif')
```

<p align="center">
  <img src="https://github.com/ivanbelenky/pywalker/blob/master/assets/grid.gif">
</p>