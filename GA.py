import random
import matplotlib.pyplot as plt
from math import sqrt

class GeneticAlgorithm:
  """
  A class representing the implementation of Genetic Algorithm to solve TSP

  Attributes:
  ----------
  pos_array : list[tuple]
    array of the coordinates (x, y) of each place in the TSP
  labels : list[str]
    array of labels for each place in the TSp
  max_gen: int
    max generations allowed in the simulation before stopping
  crossover_probability: int
    since the algorithm uses PMX, this defines the size of the middle segmentation
  mutation_probability: int
    defines how many genes (places) will be swapped in the mutation operator
  """
  

  def __init__(self, pos_array, labels, max_gen, crossover_probability, mutation_probability):
    
    self.max_gen = max_gen
    self.pos_array = pos_array
    self.labels = labels
    
    # pre-initialization parameter checking
    if crossover_probability > len(self.pos_array) - 2:
      raise ValueError(f"Crossover probability ({crossover_probability}) should allow for a 3-part segmentation.")
    
    if mutation_probability % 2 != 0:
      raise ValueError(f"Mutation probability ({mutation_probability}) should be a multiple of 2.")
    
    if mutation_probability > len(self.pos_array):
      raise ValueError(f"Mutation probability ({mutation_probability}) should not exceed the length of the pos_array ({len(pos_array)}).")

    self.crossover_probability = crossover_probability
    self.mutation_probability = mutation_probability

    # list of all stringified solutions that have been created to prevent duplications
    self.created_solutions: list[str] = []

    # holds the current best solution in the entire simulation
    self.best_solution = []

    # holds the list of fitness scores of all found best solutions  
    self.plots = []

  def _add_generated_solution(self, solution: list):
    """
    Adds the solution to the list of already created solutions
    """

    stringified = "".join(str(gene) for gene in solution)
    self.created_solutions.append(stringified)

  def _generate_random_solution(self) -> list[int]:
    """
    Returns a random order of list containing integers 0 to 9
    """
    
    _tmp = list(range(len(self.pos_array)))
    random.shuffle(_tmp)
    return _tmp
  
  def _get_distance(self, pt1, pt2):
    """
    Returns the distance of two points using the distance formula
    """

    return sqrt(
          (self.pos_array[pt2][0] - self.pos_array[pt1][0]) * (self.pos_array[pt2][0] - self.pos_array[pt1][0]) + 
          (self.pos_array[pt2][1] - self.pos_array[pt1][1]) * (self.pos_array[pt2][1] - self.pos_array[pt1][1])
          )
  
  def initialize(self):
    """
    Initialize 2 random parents and graphs
    """

    # parents 1 and 2
    self.p1 = self._generate_random_solution()
    self.p2 = self._generate_random_solution()

    self._add_generated_solution(self.p1)
    self._add_generated_solution(self.p2)

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
    """
    Main loop of the simulation
    """
    
    self.initialize()

    while self.current_gen < self.max_gen:

      # sets how often the graph for distance/gen will update
      if (self.current_gen % 20 == 0):
        self.update_plot(0, self.plots)

      # evaluate 2 parents  
      p1_score = self.fitness(self.p1)
      p2_score = self.fitness(self.p2)

      better_parent = self.p1 if p1_score < p2_score else self.p2

      # offsprings 1 and 2
      c1, c2 = self.crossover(self.p1, self.p2)

      # print("Parents: ", self.p1, self.p2)
      # print("Children: ", c1, c2)

      c1 = self.mutate(c1)
      c2 = self.mutate(c2)
      
      c1_score = self.fitness(c1)
      c2_score = self.fitness(c2)

      # get the better offspring (lower cost) and make it the parent1 for the next generation
      self.p1 = c1 if c1_score < c2_score else c2
      self._add_generated_solution(c1 if c1_score > c2_score else c2)

      if self.fitness(self.p1) < self.fitness(self.best_solution):
        # a new best solution was found
        self.best_solution = self.p1
        self.update_plot(1, self.best_solution)
        print(f"New best solution on Generation {self.current_gen}: {self.fitness(self.best_solution)}")

      self.plots.append(self.fitness(self.best_solution))

      # generate new random p2
      temp_p2 = self._generate_random_solution()

      self.current_gen += 1

      # a base condition is added here to prevent long / infinite loop 
      """
      This will iteratively generate a random solution WHILE the created solution has already been created (existing in created_solutions)
      OR the created solution's fitness score is more than 1.2x higher than the Parent 1
      AND the counter has not reached the maximum iteration (100)
      """
      count = 0
      while (("".join(str(gene) for gene in temp_p2) in self.created_solutions 
              or self.fitness(temp_p2) > (self.fitness(self.p1) * 1.2))
              and count < 100):
        temp_p2 = self._generate_random_solution()
        count += 1

      if count != 100:
        # if the loop stops before reaching the maximum iteration, then the randomly created solution is a valid Parent 2
        self.p2 = temp_p2 if self.fitness(temp_p2) < self.fitness(better_parent) else better_parent
      else:
        # 50% chance to be the better parent of the current generation and 50% to be the best solution
        if random.randint(0, 1) == 1:
          self.p2 = self.best_solution
        else:
          self.p2 = better_parent

    plt.show()
    return self.best_solution
      
  def crossover(self, parent1, parent2):
    """
    Perform PMX to the two passed parents
    """

    # get the two points that will create n-sized (crossover probability) middle segmentation
    pt1 = (len(self.pos_array) - self.crossover_probability) // 2
    pt2  = pt1 + self.crossover_probability

    # Get the middle segment of each parent
    mid1 = parent1[pt1:pt2]
    mid2 = parent2[pt1:pt2]

    # Create new children with the mid segment
    child1 = parent1[:pt1] + mid2 + parent1[pt2:]
    child2 = parent2[:pt1] + mid1 + parent2[pt2:]

    # Fixes dupes
    def fix_duplicates(child, segment, other_parent_segment):
      for idx, num in enumerate(child):
        if idx < pt1 or idx >= pt2:
          if num in segment:
            idx_other_parent = segment.index(num)
            while other_parent_segment[idx_other_parent] in segment:
              idx_other_parent = segment.index(other_parent_segment[idx_other_parent])
            child[idx] = other_parent_segment[idx_other_parent]
        
    fix_duplicates(child1, mid2, mid1)
    fix_duplicates(child2, mid1, mid2)

    return child1, child2

  def fitness(self, solution):
    """
    Calculate the fitness score of the passed solution
    """

    if len(solution) == 0:
       return float("+inf")
  
    total_distance = 0
    for i in range(len(solution)):
      if (i < len(solution) - 1):
        total_distance += self._get_distance(solution[i+1], solution[i])
      else:
        total_distance += self._get_distance(solution[i], solution[0])
    
    return total_distance

  def mutate(self, solution):
    """
    Mutates the passed solution with a defined mutation probability
    """

    for _ in range(0, self.mutation_probability, 2):
      random_pt1 = random.randint(0, len(solution) - 1)
      random_pt2 = random.randint(0, len(solution) - 1)

      while random_pt2 == random_pt1:
        random_pt2 = random.randint(0, len(solution) - 1)

      tmp = solution[random_pt1]
      solution[random_pt1] = solution[random_pt2]
      solution[random_pt2] = tmp

    return solution

  def update_plot(self, axis_idx, plots):
    """
    Updates the plot for distance/gen and best path
    """
    
    self.axis[axis_idx].cla()

    if axis_idx == 0: 
      self.axis[axis_idx].set_title("Distance of Best Solution Every Generation")
      self.axis[axis_idx].set_xlabel("Generations")
      self.axis[axis_idx].set_ylabel("Distance")
      self.axis[axis_idx].plot(plots)

    else:

      self.axis[axis_idx].set_title(f"Best Path: {plots} \nDistance: {self.fitness(plots)}")
      self.axis[axis_idx].set_xticks(range(17))
      self.axis[axis_idx].set_yticks(range(12))
      self.axis[axis_idx].grid(color='#2A3459', alpha=0.25)

      tmp_array = []
      for i in plots:
        tmp_array.append(self.pos_array[i])
      tmp_array.append(self.pos_array[plots[0]])

      self.axis[axis_idx].scatter(*zip(*tmp_array))
      self.axis[axis_idx].plot(*zip(*tmp_array))
      
      for i in range(len(plots)):
        self.axis[axis_idx].annotate(f"{self.labels[i]}({i})", self.pos_array[i], fontsize=10)

    plt.pause(0.05)

# 0 - Gate,       1 - Gym,      2 - Pool
# 3 - Oval,       4 - Obelisk,  5 - Lagoon
# 6 - Main Bldg   7 - Charlie   8 - Library
# 9 - Linear 
pos_array = [ (0, 7), (2, 9), (5, 9),
              (3, 3),  (8, 6), (10, 4),
              (12, 2),  (13, 3), (14, 5),
              (15, 0)]

labels = ["Gate",      "Gym",      "Pool",
          "Oval",      "Obelisk",  "Lagoon",
          "Main Bldg", "Charlie",  "Library",
          "Linear Park"]

max_gen = 5000
crossover_prob = 4 
mutation_prob = 4

GA = GeneticAlgorithm(pos_array, labels, max_gen, crossover_prob, mutation_prob)
bs = GA.start()
print("Best: ", bs, GA.fitness(bs))