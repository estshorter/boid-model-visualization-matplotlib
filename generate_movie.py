from model import BoidFlockers
import matplotlib.animation as anm


def update(iter):
    model.step()
    model.draw_succesive(enable_pause=False)


model = BoidFlockers(population=10, vision=15)
model.draw_initial()

ani = anm.FuncAnimation(model.fig, update, interval=20, frames=250)
ani.save("movie.gif", writer='imagemagick')
