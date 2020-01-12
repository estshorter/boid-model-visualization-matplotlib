"""
Flockers
=============================================================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from boid import Boid


class BoidFlockers(Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        population: int = 100,
        width: int = 100,
        height: int = 100,
        speed: float = 1,
        vision: float = 10,
        separation: float = 2,
        cohere: float = 0.025,
        separate: float = 0.25,
        match: float = 0.04,
    ) -> None:
        """
        Create a new Flockers model.

        Args:
            population: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            separation: What's the minimum distance each Boid will attempt to
                    keep from any other
            cohere, separate, match: factors for the relative importance of
                    the three drives.        """

        self.width = width
        self.height = height
        self.agent_pos_lst = None
        self.agent_vision_lst = None
        self.fig = None
        self.ax = None
        self.text = None

        self.population = population
        self.vision = vision
        self.speed = speed
        self.separation = separation
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, True)
        self.factors = dict(cohere=cohere, separate=separate, match=match)
        self.make_agents()
        self.running = True

    def make_agents(self) -> None:
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            velocity = np.random.random(2) * 2 - 1
            boid = Boid(
                i,
                self,
                pos,
                self.speed,
                velocity,
                self.vision,
                self.separation,
                **self.factors,
            )
            self.space.place_agent(boid, pos)
            self.schedule.add(boid)

    def step(self) -> None:
        self.schedule.step()

    def draw_initial(self) -> None:
        self.fig, self.ax = plt.subplots()
        ax = self.ax
        self.text = ax.set_title(f"t={self.schedule.time:03d}")

        self.agent_pos_lst = {}
        self.agent_vision_lst = {}

        # エージェントを手前に、visionを奥側に描画する
        for agent in self.schedule.agents:
            x, y = agent.pos
            scat = ax.scatter(x, y, c="red", s=10, zorder=2)
            self.agent_pos_lst[agent.unique_id] = scat
            c = patches.Circle(
                xy=agent.pos,
                radius=self.vision,
                ec="lightgrey",
                fc="None",
                linestyle="--",
                zorder=1,
            )
            ax.add_patch(c)
            self.agent_vision_lst[agent.unique_id] = c

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")

    def draw_succesive(self) -> None:
        self.text.set_text(f"t={self.schedule.time:03d}")
        for agent in self.schedule.agents:
            self.agent_pos_lst[agent.unique_id].set_offsets(agent.pos)
            self.agent_vision_lst[agent.unique_id].center = agent.pos
