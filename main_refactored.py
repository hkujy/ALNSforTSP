"""
Adaptive Large Neighborhood Search (ALNS) for Traveling Salesman Problem (TSP)

This module implements the ALNS algorithm to solve TSP instances.
The algorithm uses adaptive destroy and repair operators with dynamic weight adjustment.

Author: Refactored for improved readability and structure
"""

import copy
import math
import random
import time
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Configure matplotlib for Chinese font support
matplotlib.rcParams['font.family'] = 'STSong'
np.set_printoptions(linewidth=400, threshold=np.inf)


class TSPSolver:
    """
    Traveling Salesman Problem solver using Adaptive Large Neighborhood Search.
    
    Attributes:
        city_coordinates: Array of city coordinates
        distance_matrix: Distance matrix between cities
        city_count: Number of cities
    """
    
    def __init__(self, city_coordinates: np.ndarray):
        """
        Initialize TSP solver with city coordinates.
        
        Args:
            city_coordinates: Array of [x, y] coordinates for each city
        """
        self.city_coordinates = city_coordinates
        self.city_count = len(city_coordinates)
        self.distance_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate Euclidean distance matrix between all cities.
        
        Returns:
            Distance matrix with shape (city_count+1, city_count+1)
        """
        # Create matrix with +1 to handle 1-indexed cities
        distance = np.zeros([self.city_count + 1, self.city_count + 1])
        
        for i in range(1, self.city_count + 1):
            for j in range(1, self.city_count + 1):
                if i != j:
                    dx = self.city_coordinates[i - 1][0] - self.city_coordinates[j - 1][0]
                    dy = self.city_coordinates[i - 1][1] - self.city_coordinates[j - 1][1]
                    distance[i][j] = math.sqrt(dx ** 2 + dy ** 2)
        
        return distance
    
    def calculate_tour_length(self, tour: List[int]) -> float:
        """
        Calculate total distance of a tour.
        
        Args:
            tour: List of city indices representing the tour
            
        Returns:
            Total distance of the tour
        """
        total_distance = 0
        
        # Calculate distances between consecutive cities
        for i in range(self.city_count - 1):
            total_distance += self.distance_matrix[tour[i]][tour[i + 1]]
        
        # Add distance from last city back to first city
        total_distance += self.distance_matrix[tour[-1]][tour[0]]
        
        return total_distance


class DestroyOperators:
    """Collection of destroy operators for ALNS algorithm."""
    
    @staticmethod
    def random_removal(solution: List[int], city_count: int, min_remove: int = 2, max_remove: int = 6) -> Tuple[List[int], List[int]]:
        """
        Randomly remove cities from the solution.
        
        Args:
            solution: Current tour solution
            city_count: Total number of cities
            min_remove: Minimum number of cities to remove
            max_remove: Maximum number of cities to remove
            
        Returns:
            Tuple of (remaining_solution, removed_cities)
        """
        temp_solution = copy.copy(solution)
        removed_cities = []
        
        # Randomly select cities to remove
        num_to_remove = random.randint(min_remove, max_remove)
        indices_to_remove = random.sample(range(0, city_count), num_to_remove)
        
        # Store removed cities and their indices
        for idx in indices_to_remove:
            removed_cities.append(temp_solution[idx])
        
        # Remove cities in reverse order to maintain index validity
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            del temp_solution[idx]
        
        return temp_solution, removed_cities
    
    @staticmethod
    def worst_distance_removal(solution: List[int], distance_matrix: np.ndarray, 
                             min_remove: int = 2, max_remove: int = 6) -> Tuple[List[int], List[int]]:
        """
        Remove cities with the worst (highest) distance contributions.
        
        Args:
            solution: Current tour solution
            distance_matrix: Distance matrix between cities
            min_remove: Minimum number of cities to remove
            max_remove: Maximum number of cities to remove
            
        Returns:
            Tuple of (remaining_solution, removed_cities)
        """
        temp_solution = copy.copy(solution)
        city_count = len(solution)
        removed_cities = []
        
        # Calculate distance contribution for each city
        distance_contributions = np.zeros(city_count)
        
        # First city: distance from last city + distance to next city
        distance_contributions[0] = (distance_matrix[solution[-1]][solution[0]] + 
                                   distance_matrix[solution[0]][solution[1]])
        
        # Last city: distance from previous city + distance to first city
        distance_contributions[-1] = (distance_matrix[solution[-2]][solution[-1]] + 
                                    distance_matrix[solution[-1]][solution[0]])
        
        # Middle cities: distance from previous + distance to next
        for i in range(1, city_count - 1):
            distance_contributions[i] = (distance_matrix[solution[i - 1]][solution[i]] + 
                                       distance_matrix[solution[i]][solution[i + 1]])
        
        # Select worst cities to remove
        num_to_remove = random.randint(min_remove, max_remove)
        indices_to_remove = []
        
        for _ in range(num_to_remove):
            worst_idx = np.argmax(distance_contributions)
            indices_to_remove.append(worst_idx)
            removed_cities.append(temp_solution[worst_idx])
            distance_contributions[worst_idx] = 0  # Avoid selecting again
        
        # Remove cities in reverse order
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            del temp_solution[idx]
        
        return temp_solution, removed_cities


class RepairOperators:
    """Collection of repair operators for ALNS algorithm."""
    
    @staticmethod
    def greedy_insertion(partial_solution: List[int], removed_cities: List[int], 
                        distance_matrix: np.ndarray) -> Tuple[List[int], float]:
        """
        Insert removed cities using greedy best-position insertion.
        
        Args:
            partial_solution: Current partial tour
            removed_cities: Cities to be reinserted
            distance_matrix: Distance matrix between cities
            
        Returns:
            Tuple of (complete_solution, tour_length)
        """
        current_solution = copy.copy(partial_solution)
        
        for city in removed_cities:
            best_cost = float('inf')
            best_position = 0
            
            # Try inserting at each possible position
            for pos in range(len(current_solution) + 1):
                temp_route = current_solution.copy()
                temp_route.insert(pos, city)
                
                # Calculate insertion cost
                if pos == 0:
                    # Insert at beginning
                    cost = (distance_matrix[temp_route[-1]][temp_route[0]] + 
                           distance_matrix[temp_route[0]][temp_route[1]] - 
                           distance_matrix[temp_route[-1]][temp_route[1]])
                elif pos == len(current_solution):
                    # Insert at end
                    cost = (distance_matrix[temp_route[-2]][temp_route[-1]] + 
                           distance_matrix[temp_route[-1]][temp_route[0]] - 
                           distance_matrix[temp_route[-2]][temp_route[0]])
                else:
                    # Insert in middle
                    cost = (distance_matrix[temp_route[pos-1]][temp_route[pos]] + 
                           distance_matrix[temp_route[pos]][temp_route[pos+1]] - 
                           distance_matrix[temp_route[pos-1]][temp_route[pos+1]])
                
                if cost < best_cost:
                    best_cost = cost
                    best_position = pos
            
            # Insert at best position
            current_solution.insert(best_position, city)
        
        return current_solution, 0  # Tour length will be calculated separately
    
    @staticmethod
    def noisy_greedy_insertion(partial_solution: List[int], removed_cities: List[int], 
                              distance_matrix: np.ndarray, noise_factor: float = 0.1) -> Tuple[List[int], float]:
        """
        Insert removed cities using noisy greedy insertion with randomization.
        
        Args:
            partial_solution: Current partial tour
            removed_cities: Cities to be reinserted
            distance_matrix: Distance matrix between cities
            noise_factor: Factor for random noise addition
            
        Returns:
            Tuple of (complete_solution, tour_length)
        """
        current_solution = copy.copy(partial_solution)
        
        # Find maximum distance for noise calculation
        max_distance = np.max(distance_matrix)
        
        for city in removed_cities:
            best_cost = float('inf')
            best_position = 0
            
            # Try inserting at each possible position
            for pos in range(len(current_solution) + 1):
                temp_route = current_solution.copy()
                temp_route.insert(pos, city)
                
                # Calculate insertion cost with noise
                noise = max_distance * noise_factor * random.uniform(-1, 1)
                
                if pos == 0:
                    cost = (distance_matrix[temp_route[-1]][temp_route[0]] + 
                           distance_matrix[temp_route[0]][temp_route[1]] - 
                           distance_matrix[temp_route[-1]][temp_route[1]] + noise)
                elif pos == len(current_solution):
                    cost = (distance_matrix[temp_route[-2]][temp_route[-1]] + 
                           distance_matrix[temp_route[-1]][temp_route[0]] - 
                           distance_matrix[temp_route[-2]][temp_route[0]] + noise)
                else:
                    cost = (distance_matrix[temp_route[pos-1]][temp_route[pos]] + 
                           distance_matrix[temp_route[pos]][temp_route[pos+1]] - 
                           distance_matrix[temp_route[pos-1]][temp_route[pos+1]] + noise)
                
                if cost < best_cost:
                    best_cost = cost
                    best_position = pos
            
            # Insert at best position
            current_solution.insert(best_position, city)
        
        return current_solution, 0  # Tour length will be calculated separately


class ALNSParameters:
    """Configuration parameters for ALNS algorithm."""
    
    def __init__(self):
        # Algorithm parameters
        self.destroy_operators_count = 2
        self.repair_operators_count = 2
        self.segment_length = 50  # j_max
        self.max_iterations = 1000
        
        # Scoring parameters
        self.score_best_solution = 20  # theta_1
        self.score_better_solution = 12  # theta_2
        self.score_accepted_solution = 8  # theta_3
        
        # Adaptive parameters
        self.weight_decay_factor = 0.95  # alpha
        self.acceptance_probability = 0.2  # T
        self.noise_factor = 0.1  # u


class ALNSAlgorithm:
    """
    Adaptive Large Neighborhood Search algorithm implementation.
    
    This class implements the main ALNS algorithm with adaptive operator selection
    and weight adjustment mechanisms.
    """
    
    def __init__(self, tsp_solver: TSPSolver, parameters: ALNSParameters):
        """
        Initialize ALNS algorithm.
        
        Args:
            tsp_solver: TSP solver instance
            parameters: Algorithm parameters
        """
        self.tsp_solver = tsp_solver
        self.params = parameters
        self.destroy_ops = DestroyOperators()
        self.repair_ops = RepairOperators()
        
        # Initialize operator weights and probabilities
        self.destroy_weights = np.ones(parameters.destroy_operators_count, dtype=np.float64)
        self.repair_weights = np.ones(parameters.repair_operators_count, dtype=np.float64)
        self.destroy_probabilities = np.ones(parameters.destroy_operators_count) / parameters.destroy_operators_count
        self.repair_probabilities = np.ones(parameters.repair_operators_count) / parameters.repair_operators_count
        
        # Statistics tracking (for current segment)
        self.destroy_usage_count = np.zeros(parameters.destroy_operators_count)
        self.repair_usage_count = np.zeros(parameters.repair_operators_count)
        self.destroy_scores = np.zeros(parameters.destroy_operators_count)
        self.repair_scores = np.zeros(parameters.repair_operators_count)
        
        # Timing statistics (for current segment)
        self.destroy_total_time = np.zeros(parameters.destroy_operators_count)
        self.repair_total_time = np.zeros(parameters.repair_operators_count)
        
        # Cumulative statistics (for entire run)
        self.destroy_total_usage = np.zeros(parameters.destroy_operators_count)
        self.repair_total_usage = np.zeros(parameters.repair_operators_count)
        self.destroy_cumulative_time = np.zeros(parameters.destroy_operators_count)
        self.repair_cumulative_time = np.zeros(parameters.repair_operators_count)
        
        # Average timing (calculated from cumulative)
        self.destroy_avg_time = np.zeros(parameters.destroy_operators_count)
        self.repair_avg_time = np.zeros(parameters.repair_operators_count)
    
    def _select_destroy_operator(self) -> int:
        """Select destroy operator based on current probabilities."""
        cumulative_probs = np.cumsum(self.destroy_probabilities)
        random_value = random.random()
        
        if random_value == 0:
            random_value += 1e-6
        
        for i in range(len(cumulative_probs)):
            if random_value <= cumulative_probs[i]:
                return i
        return len(cumulative_probs) - 1
    
    def _select_repair_operator(self) -> int:
        """Select repair operator based on current probabilities."""
        cumulative_probs = np.cumsum(self.repair_probabilities)
        random_value = random.random()
        
        if random_value == 0:
            random_value += 1e-6
        
        for i in range(len(cumulative_probs)):
            if random_value <= cumulative_probs[i]:
                return i
        return len(cumulative_probs) - 1
    
    def _apply_destroy_operator(self, solution: List[int], operator_id: int) -> Tuple[List[int], List[int], float]:
        """
        Apply selected destroy operator and measure execution time.
        
        Returns:
            Tuple of (partial_solution, removed_cities, execution_time)
        """
        start_time = time.perf_counter()
        
        if operator_id == 0:
            result = self.destroy_ops.random_removal(solution, self.tsp_solver.city_count)
        elif operator_id == 1:
            result = self.destroy_ops.worst_distance_removal(solution, self.tsp_solver.distance_matrix)
        else:
            raise ValueError(f"Unknown destroy operator: {operator_id}")
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        return result[0], result[1], execution_time
    
    def _apply_repair_operator(self, partial_solution: List[int], removed_cities: List[int], 
                              operator_id: int) -> Tuple[List[int], float]:
        """
        Apply selected repair operator and measure execution time.
        
        Returns:
            Tuple of (complete_solution, execution_time)
        """
        start_time = time.perf_counter()
        
        if operator_id == 0:
            result, _ = self.repair_ops.greedy_insertion(
                partial_solution, removed_cities, self.tsp_solver.distance_matrix)
        elif operator_id == 1:
            result, _ = self.repair_ops.noisy_greedy_insertion(
                partial_solution, removed_cities, self.tsp_solver.distance_matrix, self.params.noise_factor)
        else:
            raise ValueError(f"Unknown repair operator: {operator_id}")
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        return result, execution_time
    
    def _update_operator_weights(self):
        """Update operator weights based on performance scores."""
        alpha = self.params.weight_decay_factor
        
        # Update destroy operator weights
        for i in range(self.params.destroy_operators_count):
            if self.destroy_usage_count[i] == 0:
                self.destroy_weights[i] *= alpha
            else:
                avg_score = self.destroy_scores[i] / self.destroy_usage_count[i]
                self.destroy_weights[i] = alpha * self.destroy_weights[i] + (1 - alpha) * avg_score
        
        # Normalize destroy probabilities
        total_destroy_weight = np.sum(self.destroy_weights)
        self.destroy_probabilities = self.destroy_weights / total_destroy_weight
        
        # Update repair operator weights
        for i in range(self.params.repair_operators_count):
            if self.repair_usage_count[i] == 0:
                self.repair_weights[i] *= alpha
            else:
                avg_score = self.repair_scores[i] / self.repair_usage_count[i]
                self.repair_weights[i] = alpha * self.repair_weights[i] + (1 - alpha) * avg_score
        
        # Normalize repair probabilities
        total_repair_weight = np.sum(self.repair_weights)
        self.repair_probabilities = self.repair_weights / total_repair_weight
        
        # Reset counters for next segment
        self.destroy_usage_count.fill(0)
        self.repair_usage_count.fill(0)
        self.destroy_scores.fill(0)
        self.repair_scores.fill(0)
        self.destroy_total_time.fill(0)
        self.repair_total_time.fill(0)
    
    def _update_timing_statistics(self):
        """Update average timing statistics for operators."""
        # Calculate average execution times from cumulative data
        for i in range(self.params.destroy_operators_count):
            if self.destroy_total_usage[i] > 0:
                self.destroy_avg_time[i] = self.destroy_cumulative_time[i] / self.destroy_total_usage[i]
        
        for i in range(self.params.repair_operators_count):
            if self.repair_total_usage[i] > 0:
                self.repair_avg_time[i] = self.repair_cumulative_time[i] / self.repair_total_usage[i]
    
    def get_timing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive timing statistics for all operators.
        
        Returns:
            Dictionary containing timing statistics
        """
        return {
            'destroy_avg_times': self.destroy_avg_time.copy(),
            'repair_avg_times': self.repair_avg_time.copy(),
            'destroy_total_times': self.destroy_cumulative_time.copy(),
            'repair_total_times': self.repair_cumulative_time.copy(),
            'destroy_usage_counts': self.destroy_total_usage.copy(),
            'repair_usage_counts': self.repair_total_usage.copy()
        }
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Solve TSP using ALNS algorithm.
        
        Returns:
            Tuple of (best_solution, best_value, convergence_history)
        """
        # Initialize solution
        current_solution = list(range(1, self.tsp_solver.city_count + 1))
        current_value = self.tsp_solver.calculate_tour_length(current_solution)
        
        best_solution = current_solution.copy()
        best_value = current_value
        convergence_history = [best_value]
        
        iteration_count = 0
        segment_count = 0
        
        print(f"Initial solution value: {best_value:.2f}")
        
        while iteration_count < self.params.max_iterations:
            iteration_count += 1
            segment_count += 1
            
            # Select operators
            destroy_op = self._select_destroy_operator()
            repair_op = self._select_repair_operator()
            
            # Update usage counters
            self.destroy_usage_count[destroy_op] += 1
            self.repair_usage_count[repair_op] += 1
            
            # Apply destroy operator
            partial_solution, removed_cities, destroy_time = self._apply_destroy_operator(current_solution, destroy_op)
            
            # Apply repair operator
            new_solution, repair_time = self._apply_repair_operator(partial_solution, removed_cities, repair_op)
            new_value = self.tsp_solver.calculate_tour_length(new_solution)
            
            # Update timing statistics
            self.destroy_total_time[destroy_op] += destroy_time
            self.repair_total_time[repair_op] += repair_time
            
            # Update cumulative statistics
            self.destroy_total_usage[destroy_op] += 1
            self.repair_total_usage[repair_op] += 1
            self.destroy_cumulative_time[destroy_op] += destroy_time
            self.repair_cumulative_time[repair_op] += repair_time
            
            # Evaluate solution and update scores
            if new_value < current_value:
                current_solution = new_solution.copy()
                current_value = new_value
                
                if new_value < best_value:
                    best_solution = new_solution.copy()
                    best_value = new_value
                    self.destroy_scores[destroy_op] += self.params.score_best_solution
                    self.repair_scores[repair_op] += self.params.score_best_solution
                    print(f"Iteration {iteration_count}: New best solution found: {best_value:.2f}")
                else:
                    self.destroy_scores[destroy_op] += self.params.score_better_solution
                    self.repair_scores[repair_op] += self.params.score_better_solution
            else:
                # Accept worse solution with probability
                if random.random() < self.params.acceptance_probability:
                    current_solution = new_solution.copy()
                    current_value = new_value
                    self.destroy_scores[destroy_op] += self.params.score_accepted_solution
                    self.repair_scores[repair_op] += self.params.score_accepted_solution
            
            convergence_history.append(best_value)
            
            # Update operator weights at end of segment
            if segment_count == self.params.segment_length:
                self._update_operator_weights()
                self._update_timing_statistics()
                segment_count = 0
                print(f"Iteration {iteration_count}: Updated operator weights")
                print(f"  Destroy probabilities: {self.destroy_probabilities.tolist()}")
                print(f"  Repair probabilities: {self.repair_probabilities.tolist()}")
                print(f"  Destroy avg times (ms): {(self.destroy_avg_time * 1000).tolist()}")
                print(f"  Repair avg times (ms): {(self.repair_avg_time * 1000).tolist()}")
        
        return best_solution, best_value, convergence_history
    
    def print_final_statistics(self):
        """Print comprehensive final statistics including timing information."""
        print("\nFinal Operator Statistics:")
        print("=" * 40)
        
        print("\nDestroy Operators:")
        operator_names = ["Random Removal", "Worst Distance Removal"]
        for i in range(self.params.destroy_operators_count):
            if i < len(operator_names):
                name = operator_names[i]
            else:
                name = f"Destroy Op {i}"
            
            print(f"  {name}:")
            print(f"    Usage count: {int(self.destroy_total_usage[i])}")
            print(f"    Total time: {self.destroy_cumulative_time[i]:.4f}s")
            if self.destroy_total_usage[i] > 0:
                avg_time = self.destroy_cumulative_time[i] / self.destroy_total_usage[i]
                print(f"    Average time: {avg_time * 1000:.3f}ms")
            print(f"    Current probability: {self.destroy_probabilities[i]:.3f}")
        
        print("\nRepair Operators:")
        operator_names = ["Greedy Insertion", "Noisy Greedy Insertion"]
        for i in range(self.params.repair_operators_count):
            if i < len(operator_names):
                name = operator_names[i]
            else:
                name = f"Repair Op {i}"
            
            print(f"  {name}:")
            print(f"    Usage count: {int(self.repair_total_usage[i])}")
            print(f"    Total time: {self.repair_cumulative_time[i]:.4f}s")
            if self.repair_total_usage[i] > 0:
                avg_time = self.repair_cumulative_time[i] / self.repair_total_usage[i]
                print(f"    Average time: {avg_time * 1000:.3f}ms")
            print(f"    Current probability: {self.repair_probabilities[i]:.3f}")


def load_tsp_data(filename: str) -> Tuple[List[str], np.ndarray]:
    """
    Load TSP data from file.
    
    Args:
        filename: Path to data file
        
    Returns:
        Tuple of (city_names, city_coordinates)
    """
    city_names = []
    city_coordinates = []
    
    try:
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        city_names.append(parts[0])
                        city_coordinates.append([float(parts[1]), float(parts[2])])
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        raise
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        raise
    
    return city_names, np.array(city_coordinates)


def plot_solution(city_names: List[str], city_coordinates: np.ndarray, 
                 solution: List[int], solution_value: float, convergence_history: List[float]):
    """
    Plot the TSP solution and convergence history.
    
    Args:
        city_names: List of city names
        city_coordinates: Array of city coordinates
        solution: Best tour solution
        solution_value: Best tour value
        convergence_history: Algorithm convergence history
    """
    # Plot convergence history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(convergence_history)
    plt.title('ALNS Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Length')
    plt.grid(True)
    
    # Plot tour
    plt.subplot(1, 2, 2)
    
    # Extract coordinates for the tour
    x_coords = []
    y_coords = []
    tour_labels = []
    
    for city_idx in solution:
        coord = city_coordinates[city_idx - 1]
        x_coords.append(coord[0])
        y_coords.append(coord[1])
        tour_labels.append(city_idx)
    
    # Close the tour
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    tour_labels.append(tour_labels[0])
    
    # Plot the tour
    plt.plot(x_coords, y_coords, '-o', linewidth=2, markersize=8)
    
    # Add city labels
    for i, (x, y, label) in enumerate(zip(x_coords[:-1], y_coords[:-1], tour_labels[:-1])):
        plt.annotate(str(label), xy=(x, y), xytext=(x + 0.3, y + 0.3), 
                    fontsize=10, ha='center', va='center')
    
    plt.title(f'Best Tour (Length: {solution_value:.2f})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()


def plot_operator_timing_statistics(alns_algorithm: ALNSAlgorithm):
    """
    Plot timing statistics for destroy and repair operators.
    
    Args:
        alns_algorithm: ALNS algorithm instance with timing data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Destroy operators timing
    destroy_names = ["Random\nRemoval", "Worst Distance\nRemoval"]
    destroy_times = []
    destroy_counts = []
    
    for i in range(alns_algorithm.params.destroy_operators_count):
        if alns_algorithm.destroy_total_usage[i] > 0:
            avg_time = alns_algorithm.destroy_cumulative_time[i] / alns_algorithm.destroy_total_usage[i]
            destroy_times.append(avg_time * 1000)  # Convert to milliseconds
        else:
            destroy_times.append(0)
        destroy_counts.append(int(alns_algorithm.destroy_total_usage[i]))
    
    bars1 = ax1.bar(destroy_names[:len(destroy_times)], destroy_times, 
                    color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_title('Average Execution Time - Destroy Operators')
    ax1.set_ylabel('Average Time (ms)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val, count in zip(bars1, destroy_times, destroy_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}ms\n({count} uses)',
                ha='center', va='bottom', fontsize=10)
    
    # Repair operators timing
    repair_names = ["Greedy\nInsertion", "Noisy Greedy\nInsertion"]
    repair_times = []
    repair_counts = []
    
    for i in range(alns_algorithm.params.repair_operators_count):
        if alns_algorithm.repair_total_usage[i] > 0:
            avg_time = alns_algorithm.repair_cumulative_time[i] / alns_algorithm.repair_total_usage[i]
            repair_times.append(avg_time * 1000)  # Convert to milliseconds
        else:
            repair_times.append(0)
        repair_counts.append(int(alns_algorithm.repair_total_usage[i]))
    
    bars2 = ax2.bar(repair_names[:len(repair_times)], repair_times, 
                    color=['lightgreen', 'orange'], alpha=0.7)
    ax2.set_title('Average Execution Time - Repair Operators')
    ax2.set_ylabel('Average Time (ms)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val, count in zip(bars2, repair_times, repair_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}ms\n({count} uses)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run ALNS algorithm on TSP instance."""
    print("ALNS TSP Solver")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading TSP data...")
        city_names, city_coordinates = load_tsp_data('data.txt')
        print(f"Loaded {len(city_names)} cities")
        
        # Initialize solver
        tsp_solver = TSPSolver(city_coordinates)
        
        # Configure algorithm parameters
        params = ALNSParameters()
        params.max_iterations = 1000
        params.segment_length = 50
        
        # Initialize and run ALNS
        alns = ALNSAlgorithm(tsp_solver, params)
        
        print("\nRunning ALNS algorithm...")
        start_time = time.time()
        
        best_solution, best_value, convergence_history = alns.solve()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Print results
        print(f"\nResults:")
        print(f"Best solution: {best_solution}")
        print(f"Best tour length: {best_value:.2f}")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        
        # Print detailed operator timing statistics
        alns.print_final_statistics()
        
        # Plot results
        plot_solution(city_names, city_coordinates, best_solution, best_value, convergence_history)
        
        # Plot timing statistics
        plot_operator_timing_statistics(alns)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
