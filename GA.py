import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
import numpy as np
from math import sqrt
from time import sleep

class GeneticAlgorithm:

  def __init__(self, pos_array: list[tuple], labels, max_gen: int, 
               crossover_probability: int, mutation_probability: int):
    
    self.max_gen = max_gen
    self.pos_array = pos_array
    self.labels = labels
    self.crossover_probability = crossover_probability
    self.mutation_probability = mutation_probability

    # list of all stringified solutions that 
    # have been created to prevent duplications
    self.created_chromosomes: list[str] = []

    # holds the current best solution in the entire simulation
    self.best_solution = []
    self.plots = []

    self.figure = None
    self.axis = None

  def _add_generated_chromosome(self, chromosome: list):
    """Adds the chromosome to the list of already created chromosomes"""
    stringified = "".join(str(gene) for gene in chromosome)
    self.created_chromosomes.append(stringified)

  def _generate_random_chromosome(self) -> list[int]:
    """Returns a random order of list containing integers 0 to 9"""
    _tmp = list(range(len(self.pos_array)))
    random.shuffle(_tmp)
    return _tmp
  
  def _show_graph(self, chromosome):
    tmp_array = []
    for i in chromosome:
      tmp_array.append(self.pos_array[i])
    tmp_array.append(self.pos_array[chromosome[0]])

    plt.style.use("seaborn-v0_8-dark-palette")
    plt.xticks(range(17))
    plt.yticks(range(12))
    plt.grid(color='#2A3459', alpha=0.25)
    plt.scatter(*zip(*tmp_array))
    plt.plot(*zip(*tmp_array))
    
    for i in range(len(chromosome)):
      txt = plt.annotate(self.labels[i], self.pos_array[i], fontsize=10)
      txt.set_alpha(0.5)

    plt.pause(0.05)
    plt.clf()

  def initialize(self):
    """Initialize 2 random parents and starting parameters"""

    # parents 1 and 2
    self.p1 = self._generate_random_chromosome()
    self.p2 = self._generate_random_chromosome()

    self._add_generated_chromosome(self.p1)
    self._add_generated_chromosome(self.p2)

    # gen counter
    self.current_gen = 0

    plt.style.use("seaborn-v0_8-dark-palette")
    self.figure, self.axis = plt.subplots(1, 2, figsize=(16, 7))
    self.axis[0].set_title("Distance of Best Solution Every Generation")
    self.axis[0].set_xlabel("Generations")
    self.axis[0].set_ylabel("Distance")

    self.axis[1].set_title("Best Path")
    self.axis[1].set_xticks(range(17))
    self.axis[1].set_yticks(range(12))
    self.axis[1].grid(color='#2A3459', alpha=0.25)

  def start(self):
    """Main loop of the simulation"""
    
    self.initialize()

    while self.current_gen < self.max_gen:

      if (self.current_gen % 20 == 0):
        self.update_plot(0, self.plots)

      # evaluate 2 parents  
      p1_score = self.fitness(self.p1)
      p2_score = self.fitness(self.p2)

      better_parent = self.p1 if p1_score < p2_score else self.p2

      # offsprings 1 and 2
      c1, c2 = self.crossover(self.p1, self.p2)

      c1 = self.mutate(c1)
      c2 = self.mutate(c2)
      
      c1_score = self.fitness(c1)
      c2_score = self.fitness(c2)

      # get the better offspring (lower cost) and make it 
      # the parent1 for the next generation
      self.p1 = c1 if c1_score < c2_score else c2

      if self.fitness(self.p1) < self.fitness(self.best_solution):
        # a new best solution was found
        self.best_solution = self.p1
        self.update_plot(1, self.best_solution)
        print(self.fitness(self.best_solution))

      self.plots.append(self.fitness(self.best_solution))

      # generate new random p2
      temp_p2 = self._generate_random_chromosome()

      self.current_gen += 1
      count = 0
      while (("".join(str(gene) for gene in temp_p2) in self.created_chromosomes 
              or self.fitness(temp_p2) > (self.fitness(self.p1) + self.fitness(self.p1) * .5))
              and count < 100):
        temp_p2 = self._generate_random_chromosome()
        count += 1

      if count != 100:
        self.p2 = temp_p2 if self.fitness(temp_p2) < self.fitness(better_parent) else better_parent
      else:
        self.p2 = self.best_solution

    return self.best_solution
      
  def select(self):
    pass

  def crossover(self, p1, p2):
    # Get the middle segment of each parent
    mid1 = p1[3:7]
    mid2 = p2[3:7]

    # Create new children with the mid segment
    child1 = p1[:3] + mid2 + p1[7:]
    child2 = p2[:3] + mid1 + p2[7:]

    # Fixes dupes
    def fix_duplicates(child, segment, other_parent_segment):
      for idx, num in enumerate(child):
        if idx < 3 or idx >= 7:
          if num in segment:
            idx_other_parent = segment.index(num)
            while other_parent_segment[idx_other_parent] in segment:
              idx_other_parent = segment.index(other_parent_segment[idx_other_parent])
            child[idx] = other_parent_segment[idx_other_parent]
        
    fix_duplicates(child1, mid2, mid1)
    fix_duplicates(child2, mid1, mid2)

    return child1, child2

  def fitness(self, chromosome):

    if len(chromosome) == 0:
       return float("+inf")
  
    total_distance = 0
    for i in range(len(chromosome)):
      if (i < len(chromosome) - 1):
        total_distance += sqrt(
          (self.pos_array[chromosome[i+1]][0] - self.pos_array[chromosome[i]][0]) * (self.pos_array[chromosome[i+1]][0] - self.pos_array[chromosome[i]][0]) + 
          (self.pos_array[chromosome[i+1]][1] - self.pos_array[chromosome[i]][1]) * (self.pos_array[chromosome[i+1]][1] - self.pos_array[chromosome[i]][1])
          )
      else:
        total_distance += sqrt(
          (self.pos_array[chromosome[i]][0] - self.pos_array[chromosome[0]][0]) * (self.pos_array[chromosome[i]][0] - self.pos_array[chromosome[0]][0]) + 
          (self.pos_array[chromosome[i]][1] - self.pos_array[chromosome[0]][1]) * (self.pos_array[chromosome[i]][1] - self.pos_array[chromosome[0]][1])
          )
    
    return total_distance

  def mutate(self, chromosome):

    for i in range(2):
      random_pt1 = random.randint(0, len(chromosome) - 1)
      random_pt2 = random.randint(0, len(chromosome) - 1)

      tmp = chromosome[random_pt1]
      chromosome[random_pt1] = chromosome[random_pt2]
      chromosome[random_pt2] = tmp

    return chromosome

  def update_plot(self, axis_idx, plots):
    
    self.axis[axis_idx].cla()

    if axis_idx == 0: 
      self.axis[axis_idx].set_title("Distance of Best Solution Every Generation")
      self.axis[axis_idx].set_xlabel("Generations")
      self.axis[axis_idx].set_ylabel("Distance")
      self.axis[axis_idx].plot(plots)

    else:

      self.axis[axis_idx].set_title(f"Best Path (so far)\nDistance: {self.fitness(plots)}")
      self.axis[axis_idx].set_xticks(range(17))
      self.axis[axis_idx].set_yticks(range(12))
      self.axis[axis_idx].grid(color='#2A3459', alpha=0.25)

      tmp_array = []
      for i in plots:
        tmp_array.append(self.pos_array[i])
      tmp_array.append(self.pos_array[plots[0]])

      # self.axis[axis_idx].style.use("seaborn-v0_8-dark-palette")
      self.axis[axis_idx].scatter(*zip(*tmp_array))
      self.axis[axis_idx].plot(*zip(*tmp_array))
      
      for i in range(len(plots)):
        txt = self.axis[axis_idx].annotate(self.labels[i], self.pos_array[i], fontsize=10)
        txt.set_alpha(0.5)

    plt.pause(0.05)

# 0 - Gate,       1 - Gym,      2 - Pool
# 3 - Oval,       4 - Obelisk,  5 - Lagoon
# 6 - Main Bldg   7 - Charlie   8 - Library
# 9 - Linear 
pos_array = [ (0, 10), (1, 7), (2, 5),
              (7, 8),  (7, 6), (10, 0),
              (9, 7),  (12, 9), (15, 3),
              (14, 8)]

labels = ["Gate",      "Gym",      "Pool",
          "Oval",      "Obelisk",  "Lagoon",
          "Main Bldg", "Charlie",  "Library",
          "Linear Park"]

# pos_array = [ (random.randint(0, 100), random.randint(0, 100)) for _ in range(100) ]

max_gen = 5000
crossover_prob = 4 
mutation_prob = 2

GA = GeneticAlgorithm(pos_array, labels, max_gen, crossover_prob, mutation_prob)
bs = GA.start()

print("Best: ", bs, GA.fitness(bs))