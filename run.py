from model import BoidFlockers

max_iter = 1000

model = BoidFlockers(population=10, vision=15)
model.draw_initial()
for iter in range(max_iter):
    model.step()
    model.draw_succesive()
