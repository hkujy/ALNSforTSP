# Code Refactoring Summary

## Overview
The original `main.py` has been refactored into `main_refactored.py` with significant improvements in code structure, readability, and maintainability.

## Key Improvements

### 1. Object-Oriented Design
- **Before**: Procedural code with global variables and functions
- **After**: Well-structured classes with clear responsibilities:
  - `TSPSolver`: Handles TSP-specific calculations
  - `DestroyOperators`: Collection of destroy operations
  - `RepairOperators`: Collection of repair operations
  - `ALNSParameters`: Configuration management
  - `ALNSAlgorithm`: Main algorithm implementation

### 2. Type Hints and Documentation
- **Before**: No type hints, minimal documentation
- **After**: Comprehensive type hints using `typing` module, detailed docstrings for all classes and methods

### 3. Code Organization
- **Before**: All code in a single file with mixed concerns
- **After**: Logical separation of concerns:
  - Data loading and preprocessing
  - Algorithm components
  - Visualization
  - Main execution logic

### 4. Improved Naming Conventions
- **Before**: Mixed naming conventions (e.g., `destory_operators`, `city_number`)
- **After**: Consistent Python naming conventions (e.g., `destroy_operators`, `city_count`)

### 5. Error Handling
- **Before**: No error handling for file operations
- **After**: Proper exception handling with informative error messages

### 6. Configuration Management
- **Before**: Hardcoded parameters scattered throughout the code
- **After**: Centralized configuration in `ALNSParameters` class

### 7. Algorithm Structure
- **Before**: Monolithic function with nested logic
- **After**: Modular design with separate methods for:
  - Operator selection
  - Weight updates
  - Solution evaluation
  - Statistics tracking

### 8. Code Readability
- **Before**: Mixed languages in comments (Chinese/English), unclear variable names
- **After**: Clear English documentation, descriptive variable names, consistent formatting

## Specific Technical Improvements

### Distance Matrix Calculation
```python
# Before: Nested loops with unclear indexing
for i in range(1, city_number + 1):
    for j in range(1, city_number + 1):
        distance[i][j] = math.sqrt((city_condition[i - 1][0] - city_condition[j - 1][0]) ** 2 + ...)

# After: Clear method with proper variable names
def _calculate_distance_matrix(self) -> np.ndarray:
    distance = np.zeros([self.city_count + 1, self.city_count + 1])
    for i in range(1, self.city_count + 1):
        for j in range(1, self.city_count + 1):
            if i != j:
                dx = self.city_coordinates[i - 1][0] - self.city_coordinates[j - 1][0]
                dy = self.city_coordinates[i - 1][1] - self.city_coordinates[j - 1][1]
                distance[i][j] = math.sqrt(dx ** 2 + dy ** 2)
```

### Operator Selection
```python
# Before: Nested if-else statements with unclear logic
temp_D = np.cumsum(P_destory)
temp_probability_D = np.random.rand()
if temp_probability_D == 0:
    temp_probability_D += 0.000001
for i in range(destory_size):
    if i == 0:
        if 0 < temp_probability_D <= temp_D[0]:
            destory_number = i
    else:
        if temp_D[i-1] < temp_probability_D <= temp_D[i]:
            destory_number = i

# After: Clean method with clear logic
def _select_destroy_operator(self) -> int:
    cumulative_probs = np.cumsum(self.destroy_probabilities)
    random_value = random.random()
    
    if random_value == 0:
        random_value += 1e-6
    
    for i in range(len(cumulative_probs)):
        if random_value <= cumulative_probs[i]:
            return i
    return len(cumulative_probs) - 1
```

### Parameter Management
```python
# Before: Scattered parameter definitions
destory_size = 2
repair_size = 2
j_max = 50
iterations = j_max * 20
theta_1 = 20
theta_2 = 12
theta_3 = 8
alpha = 0.95
T = 0.2
u = 0.1

# After: Centralized configuration class
class ALNSParameters:
    def __init__(self):
        self.destroy_operators_count = 2
        self.repair_operators_count = 2
        self.segment_length = 50
        self.max_iterations = 1000
        self.score_best_solution = 20
        self.score_better_solution = 12
        self.score_accepted_solution = 8
        self.weight_decay_factor = 0.95
        self.acceptance_probability = 0.2
        self.noise_factor = 0.1
```

## Benefits of Refactoring

1. **Maintainability**: Easier to modify and extend individual components
2. **Testability**: Each class and method can be tested independently
3. **Reusability**: Components can be reused in other projects
4. **Readability**: Code is self-documenting with clear structure
5. **Debugging**: Easier to locate and fix issues
6. **Scalability**: Easy to add new operators or modify existing ones

## Usage Example

```python
# Load data
city_names, city_coordinates = load_tsp_data('data.txt')

# Initialize solver
tsp_solver = TSPSolver(city_coordinates)

# Configure parameters
params = ALNSParameters()
params.max_iterations = 1000

# Run algorithm
alns = ALNSAlgorithm(tsp_solver, params)
best_solution, best_value, history = alns.solve()

# Visualize results
plot_solution(city_names, city_coordinates, best_solution, best_value, history)
```

The refactored code maintains all the original functionality while providing a much cleaner, more maintainable, and extensible codebase.
