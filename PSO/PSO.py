import random
import time
import matplotlib.pyplot as plt

def objective_function(O):
    x = O[0]
    y = O[1]
    nonlinear_constraint = (x - 1) ** 3 - y + 1
    linear_constraint = x + y - 2
    penalty1 = 1 if nonlinear_constraint > 0 else 0
    penalty2 = 1 if linear_constraint > 0 else 0
    z = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 + penalty1 + penalty2
    return z

bounds = [(-1.5, 1.5), (-0.5, 2.5)]  # upper and lower bounds of variables
nv = 2  # number of variables
mm = -1  # if minimization problem, mm = -1; if maximization problem, mm = 1

# PARAMETERS OF PSO
particle_size = 120  # number of particles
iterations = 200  # max number of iterations
w = 0.8  # inertia constant
c1 = 1  # cognitive constant
c2 = 2  # social constant

# Visualization
fig = plt.figure()
ax = fig.add_subplot()
plt.title('Evolutionary process of the objective function value')
plt.xlabel("Iteration")
plt.ylabel("Objective function")

initial_fitness = float('inf') if mm == -1 else -float('inf')

# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
        self.particle_position = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(nv)]
        self.particle_velocity = [random.uniform(-1, 1) for i in range(nv)]
        self.local_best_particle_position = list(self.particle_position)
        self.fitness_local_best_particle_position = initial_fitness
        self.fitness_particle_position = initial_fitness

    def evaluate(self, objective_function):
        self.fitness_particle_position = objective_function(self.particle_position)
        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = list(self.particle_position)
                self.fitness_local_best_particle_position = self.fitness_particle_position
        else:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = list(self.particle_position)
                self.fitness_local_best_particle_position = self.fitness_particle_position

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] += self.particle_velocity[i]
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]

class PSO:
    def __init__(self, objective_function, bounds, particle_size, iterations):
        global_best_particle_position = []
        fitness_global_best_particle_position = initial_fitness
        swarm_particle = [Particle(bounds) for _ in range(particle_size)]
        A = []

        for i in range(iterations):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)
                if mm == -1:
                    if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = swarm_particle[j].fitness_particle_position
                else:
                    if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = swarm_particle[j].fitness_particle_position

            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)

            A.append(fitness_global_best_particle_position)  # record the best fitness
            # Visualization
            ax.plot(A, color="r")
            fig.canvas.draw()
            ax.set_xlim(left=max(0, i - iterations), right=i + 3)
            time.sleep(0.001)

        print('RESULT:')
        print('Optima solution:', global_best_particle_position)
        print('Objective function value:', fitness_global_best_particle_position)

PSO(objective_function, bounds, particle_size, iterations)
plt.show()
                