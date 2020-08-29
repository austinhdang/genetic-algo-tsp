import random
import sys
import math
import numpy as np
import time
from multiprocessing import Process


def read_coordinates(coordinate_file):
    """ Reads coordinates from file and creates coordinates list """
    coordinates = []
    for line in coordinate_file:
        node_id, x, y = line.split(" ")
        coordinates.append((float(x), float(y)))
    return coordinates


def read_nodes(coordinate_file):
    """ Reads nodes from file and creates nodes list """
    nodes = []
    for line in coordinate_file:
        node_id, x, y = line.split(" ")
        nodes.append(node_id)
    return nodes


class City:
    def __init__(self, x, y, node_id):
        self.x = x
        self.y = y
        self.node_id = node_id

    def get_distance(self, city):
        dx = abs(self.x - city.x)
        dy = abs(self.y - city.y)
        distance = math.sqrt(dx ** 2 + dy ** 2)
        distance = int(round(distance))
        return distance

    def get_node(self):
        return self.node_id


class Fitness:
    def __init__(self, tour):
        self.tour = tour
        self.distance = 0
        self.fitness = 0.0

    def tour_distance(self):
        if self.distance == 0:
            tour_cost = 0
            for i in range(len(self.tour)):
                starting_city = self.tour[i]
                ending_city = None
                if i < len(self.tour) and i != len(self.tour) - 1:
                    ending_city = self.tour[i + 1]
                else:
                    ending_city = self.tour[0]
                tour_cost += starting_city.get_distance(ending_city)
            self.distance = tour_cost
        return self.distance

    def tour_fitness(self):
        if self.fitness == 0:
            self.distance = self.tour_distance()
            self.fitness = 1 / self.distance
        return self.fitness


def generate_random_tour(cities):
    tour = random.sample(cities, len(cities))
    return tour


def create_first_generation(population_size, cities):
    population = []
    for i in range(population_size):
        population.append(generate_random_tour(cities))
    return population


def rank_tours(population):
    fitness_matrix = {}
    for i in range(len(population)):
        fitness_matrix[i] = Fitness(population[i]).tour_fitness()
    ranked_tours = sorted(
        fitness_matrix.items(), key=lambda fitness: fitness[1], reverse=True
    )
    return ranked_tours


def create_mating_pool(population, ranked_tours, num_elites):
    mating_pool = []
    selected_tours = []
    cumulative_sum = []
    relative_weight = []
    array = np.array(ranked_tours)
    for i in range(len(ranked_tours)):
        cumulative_sum.append(array[i][1].cumsum())
        relative_weight.append(cumulative_sum[i] / array[i][1].sum())
    for i in range(num_elites):
        selected_tours.append(ranked_tours[i][0])
    for i in range(len(ranked_tours) - num_elites):
        drawn_number = random.random()
        for i in range(len(ranked_tours)):
            if drawn_number <= relative_weight[i]:
                selected_tours.append(ranked_tours[i][0])
                break
    for i in range(len(selected_tours)):
        index = selected_tours[i]
        mating_pool.append(population[index])
    return mating_pool


def ordered_crossover(parent1, parent2):
    offspring = []
    offspring_subset1 = []
    offspring_subset2 = []
    gene_A = random.randint(0, len(parent1) - 1)
    gene_B = random.randint(0, len(parent1) - 1)
    if gene_A == gene_B:
        while gene_A == gene_B:
            gene_B = random.randint(0, len(parent1) - 1)
    if gene_A < gene_B:
        start_gene = gene_A
        end_gene = gene_B
    else:
        start_gene = gene_B
        end_gene = gene_A
    for i in range(start_gene, end_gene):
        offspring_subset1.append(parent1[i])
    for gene in parent2:
        if gene not in offspring_subset1:
            offspring_subset2.append(gene)
    offspring = offspring_subset1 + offspring_subset2
    return offspring


def crossover_population(mating_pool, num_elites):
    offsprings = []
    for i in range(num_elites):
        offsprings.append(mating_pool[i])
    random.shuffle(mating_pool)
    for i in range(len(mating_pool) - num_elites):
        offspring = ordered_crossover(
            mating_pool[i], mating_pool[len(mating_pool) - 1 - i]
        )
        offsprings.append(offspring)
    return offsprings


def mutate_individual(individual, mutation_probability):
    for swap_index1 in range(len(individual)):
        if mutation_probability > random.random():
            swap_index2 = random.randint(0, len(individual) - 1)
            city1 = individual[swap_index1]
            city2 = individual[swap_index2]
            individual[swap_index1] = city2
            individual[swap_index2] = city1
    return individual


def mutate_population(population, mutation_probability):
    mutated_population = []
    for individual in range(len(population)):
        mutated_individual = mutate_individual(
            population[individual], mutation_probability
        )
        mutated_population.append(mutated_individual)
    return mutated_population


def create_next_generation(population, num_elites, mutation_probability):
    ranked_tours = rank_tours(population)
    mating_pool = create_mating_pool(population, ranked_tours, num_elites)
    offsprings = crossover_population(mating_pool, num_elites)
    next_generation = mutate_population(offsprings, mutation_probability)
    return next_generation


def genetic_algorithm(
    population, population_size, num_elites, mutation_probability, num_generations
):
    tour = ""
    current_population = create_first_generation(population_size, population)
    ranked_first_tours = rank_tours(current_population)
    print("Generation 1 distance: " + str(int(1 / ranked_first_tours[0][1])))
    for i in range(num_generations):
        current_population = create_next_generation(
            current_population, num_elites, mutation_probability
        )
        ranked_current_tours = rank_tours(current_population)
        print(
            "Generation "
            + str(i)
            + " distance: "
            + str(int(1 / ranked_current_tours[0][1]))
        )
    print("Final distance: " + str(int(1 / ranked_current_tours[0][1])))
    best_tour_index = ranked_current_tours[0][0]
    best_tour = current_population[best_tour_index]
    for city in best_tour:
        tour += str(city.get_node()) + " "
    tour += str(best_tour[0].get_node())
    print(tour)
    f = open("output-tour.txt", "w+")
    f.write(str(int(1 / ranked_current_tours[0][1])) + "\n")
    f.write(tour)
    return


def main():
    input_file = open(sys.argv[1], "r")
    coordinates = read_coordinates(input_file)
    input_file = open(sys.argv[1], "r")
    nodes = read_nodes(input_file)
    cities = []
    for i in range(len(coordinates)):
        cities.append(City(x=coordinates[i][0], y=coordinates[i][1], node_id=nodes[i]))
    genetic_algorithm(
        population=cities,
        population_size=100,
        num_elites=5,
        mutation_probability=0.015,
        num_generations=1000,
    )
    print("Time elapsed: " + str(time.process_time()) + " seconds")


if __name__ == "__main__":
    p = Process(target=main)
    p.start()
    p.join(timeout=sys.argv[3])
    p.terminate()

