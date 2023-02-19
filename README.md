# Walker

Something can emerge from this. At the moment just random walkish behavior. 

[leal](https://github.com/ivanbelenky/pywalker/blob/master/assets/graph_walker.png)

```python
env = simpy.Environment()
walker = Walker(env, random_decider, 3)
env.process(walker.run())
env.run(until=1000)
```
