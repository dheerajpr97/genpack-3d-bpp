import random
from copy import deepcopy

import numpy as np
from loguru import logger

from src import config
from src.models import layers
from src.models.superitems import SuperitemPool
from src.utils import utils
from src.utils.visualization_ga import (
    visualize_chromosome,
    visualize_constraint_violations,
    visualize_crossover,
    visualize_fitness_evolution,
    visualize_initialization_strategies,
    visualize_mutation,
    visualize_population,
)

# GA Parameters
POPULATION_SIZE = 100
GENERATIONS = 50
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.2
ELITE_COUNT = 4


def optimize_residuals(
    not_covered_superitems, first_phase_items, pallet_dims, base_z_level, visualization_options=None
):
    """Optimize residual items using genetic algorithm."""
    # Default visualization options
    if visualization_options is None:
        visualization_options = {"enable": False}

    # Extract visualization options
    viz_enabled = visualization_options.get("enable", False)
    viz_init = viz_enabled and visualization_options.get("init", False)
    viz_evolution = viz_enabled and visualization_options.get("evolution", False)
    viz_operations = viz_enabled and visualization_options.get("operations", False)
    viz_interval = visualization_options.get(
        "viz_interval", 5
    )  # Default: show every 10 generations

    logger.info(f"Starting GA for {len(not_covered_superitems)} residual superitems")
    if not not_covered_superitems:
        logger.warning("No residual superitems to optimize")
        return layers.LayerPool(SuperitemPool(), pallet_dims)

    # Track fitness history for visualization
    fitness_history = []

    # Initialize population with enhanced chromosome representation
    population = initialize_population(not_covered_superitems, first_phase_items, pallet_dims)
    logger.info(f"Initial population size: {len(population)}")

    # Plot initial population if enabled
    if viz_enabled:
        visualize_population(population=population, pallet_dims=pallet_dims)

    # Plot initial initialization strategies if enabled
    if viz_init:
        visualize_initialization_strategies(population=population, pallet_dims=pallet_dims)

    # Initial evaluation with enhanced fitness function
    logger.debug(f"Base z-level at GA input: {base_z_level}")
    fitness_scores = [
        fitness_function(chrom, not_covered_superitems, pallet_dims, base_z_level)
        for chrom in population
    ]
    logger.info(f"Initial best fitness score: {np.max(fitness_scores)}")

    # Record initial fitness data
    fitness_history.append(
        {
            "generation": 0,
            "best": max(fitness_scores),
            "average": sum(fitness_scores) / len(fitness_scores),
            "worst": min(fitness_scores),
        }
    )

    # Add defensive check for empty population or all zero fitness scores
    if not population or not fitness_scores or max(fitness_scores) == 0:
        logger.warning("Empty population or all zero fitness scores")
        return layers.LayerPool(SuperitemPool(), pallet_dims)

    best_solution = deepcopy(population[np.argmax(fitness_scores)])
    best_score = max(fitness_scores)

    no_improvement_count = 0

    # Track if we've visualized operations in the current interval
    viz_crossover_done = False
    viz_mutation_done = False

    # Main GA loop with improved parameters
    for generation in range(GENERATIONS):
        # Reset visualization flags at interval boundaries
        if generation % viz_interval == 0:
            viz_crossover_done = False
            viz_mutation_done = False

        # More elites for better convergence
        elite_count = max(4, int(len(population) * 0.1))
        elite_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )[:elite_count]
        elite_solutions = [deepcopy(population[i]) for i in elite_indices]

        # Selection and reproduction
        new_population = []

        # Add elites directly
        new_population.extend(elite_solutions)

        # Create the rest of the population
        while len(new_population) < POPULATION_SIZE:
            if not fitness_scores or sum(fitness_scores) == 0:
                # If all fitness scores are zero, select randomly
                parents = (
                    random.sample(population, 2)
                    if len(population) >= 2
                    else [population[0], population[0]]
                )
            else:
                # Use adaptive tournament selection based on generation
                tournament_ratio = 0.1 + (
                    0.3 * generation / GENERATIONS
                )  # Gradually increase selection pressure
                tournament_size = max(3, int(POPULATION_SIZE * tournament_ratio))

                # Ensure tournament size doesn't exceed population size
                tournament_size = min(tournament_size, len(population))

                # Select first parent
                tournament1 = random.sample(list(range(len(population))), tournament_size)
                parent1_idx = max(tournament1, key=lambda idx: fitness_scores[idx])

                # Select second parent ensuring it's different from first
                if len(population) > 1:
                    # Create second tournament excluding first parent
                    available_for_tournament2 = [
                        i for i in range(len(population)) if i != parent1_idx
                    ]
                    tournament_size2 = min(tournament_size, len(available_for_tournament2))
                    tournament2 = random.sample(available_for_tournament2, tournament_size2)
                    parent2_idx = max(tournament2, key=lambda idx: fitness_scores[idx])
                else:
                    # Edge case: only one chromosome in population
                    parent2_idx = parent1_idx

                parents = [population[parent1_idx], population[parent2_idx]]

            # Apply crossover with adaptive probability
            crossover_prob = 0.5 * (
                1.0 + 0.1 * generation / GENERATIONS
            )  # Increases slightly over time
            should_visualize_crossover = (
                viz_operations and not viz_crossover_done and generation % viz_interval == 0
            )

            if random.random() < crossover_prob:
                # Store parents for visualization
                parent1, parent2 = deepcopy(parents[0]), deepcopy(parents[1])
                child1, child2 = crossover(parents[0], parents[1])

                # Visualize crossover once per interval
                if should_visualize_crossover:
                    visualize_crossover(parent1, parent2, child1, child2, pallet_dims)
                    viz_crossover_done = True
            else:
                child1, child2 = deepcopy(parents[0]), deepcopy(parents[1])

            # Apply mutation with probability that decreases with generation
            base_mutation_prob = 0.35
            current_mutation_prob = base_mutation_prob * (1.0 - 0.5 * generation / GENERATIONS)
            should_visualize_mutation = (
                viz_operations and not viz_mutation_done and generation % viz_interval == 0
            )

            # Force mutation if we need to visualize it (guarantees visualization)
            force_mutation = should_visualize_mutation

            # Apply mutation more often early in the run
            if random.random() < current_mutation_prob or force_mutation:
                # Store chromosome before mutation for visualization
                before_mutation = deepcopy(child1)
                mutation_type = random.choice(
                    [
                        "center_tall_items",
                        "fill_holes",
                        "compact_arrangement",
                        "optimize_contact",
                        "improve_support",
                        "balance_center",
                        "swap_items",
                        "swap_layers",
                        "add_remove_layer",
                    ]
                )
                child1 = mutate(child1, not_covered_superitems, pallet_dims)

                # Visualize mutation once per interval if it actually changes something
                if should_visualize_mutation and child1 != before_mutation:
                    visualize_mutation(before_mutation, child1, pallet_dims, mutation_type)
                    viz_mutation_done = True

            if random.random() < current_mutation_prob:
                # Store chromosome before mutation for visualization
                before_mutation = deepcopy(child2)
                mutation_type = random.choice(
                    [
                        "center_tall_items",
                        "fill_holes",
                        "compact_arrangement",
                        "optimize_contact",
                        "improve_support",
                        "balance_center",
                        "swap_items",
                        "swap_layers",
                        "add_remove_layer",
                    ]
                )
                child2 = mutate(child2, not_covered_superitems, pallet_dims)

                # Only visualize if we haven't already done so for this interval
                if (
                    should_visualize_mutation
                    and not viz_mutation_done
                    and child2 != before_mutation
                ):
                    visualize_mutation(before_mutation, child2, pallet_dims, mutation_type)
                    viz_mutation_done = True

            # Only add children if there's room
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        # Keep population size consistent
        population = new_population[:POPULATION_SIZE]

        # Evaluate new population
        fitness_scores = [
            fitness_function(chrom, not_covered_superitems, pallet_dims, base_z_level)
            for chrom in population
        ]

        # Record fitness data
        fitness_history.append(
            {
                "generation": generation + 1,
                "best": max(fitness_scores),
                "average": sum(fitness_scores) / len(fitness_scores),
                "worst": min(fitness_scores),
            }
        )

        # Defensive check - ensure we have valid fitness scores
        if not fitness_scores:
            logger.warning(f"No valid fitness scores in generation {generation+1}")
            continue

        # Track best solution
        current_best_idx = np.argmax(fitness_scores)
        current_best_score = fitness_scores[current_best_idx]

        # Check for improvement with a small tolerance to recognize minor improvements
        improvement_tolerance = 0.001  # Smaller tolerance from former implementation
        if current_best_score > best_score + improvement_tolerance:
            best_score = current_best_score
            best_solution = deepcopy(population[current_best_idx])
            no_improvement_count = 0
            logger.info(f"Generation {generation+1}: New best score = {best_score:.4f}")
        else:
            no_improvement_count += 1
            logger.debug(
                f"Generation {generation+1}: No improvement (count: {no_improvement_count})"
            )

        # Early stopping if no improvement for several GENERATIONS, but wait longer
        if no_improvement_count >= 20:
            logger.info(f"Stopping early after {generation+1} GENERATIONS due to no improvement")
            break

    # Defensive check - make sure we have a valid solution before conversion
    if not best_solution:
        logger.warning("No valid solution found")
        return layers.LayerPool(SuperitemPool(), pallet_dims)

    # Convert best solution to LayerPool with coordinates directly assigned during evaluation
    optimized_pool = convert_to_layer_pool(best_solution, pallet_dims, base_z_level)

    # Log results
    items_packed = sum(len(layer["items"]) for layer in best_solution) if best_solution else 0
    logger.info(
        f"GA completed: {len(best_solution)} layers with {items_packed} items, fitness = {best_score:.4f}"
    )

    # Visualize fitness evolution if enabled
    if viz_evolution:
        visualize_fitness_evolution(fitness_history)

    # Always visualize final solution
    # visualize_chromosome(best_solution, pallet_dims, title="Final Best Solution")

    # Also visualize constraint violations if needed
    # if viz_enabled:
    #     visualize_constraint_violations(best_solution, pallet_dims, title="Final Solution Constraint Check")

    return optimized_pool


def initialize_population(
    not_covered_superitems, first_phase_items, pallet_dims, population_size=POPULATION_SIZE
):
    """Initialize GA population with diverse packing strategies."""
    population = []
    stacking_surfaces = []
    # Get highest z-level from first phase items
    max_z = 0
    if first_phase_items and isinstance(first_phase_items, layers.LayerPool):
        for layer in first_phase_items:
            items_coords = layer.get_items_coords()
            items_dims = layer.get_items_dims()
            for item_id, coord in items_coords.items():
                if item_id in items_dims:
                    dims = items_dims[item_id]
                    max_z = max(max_z, coord.z + dims.height)
                    # Also create stacking surfaces
                    stacking_surfaces.append(
                        {
                            "x": coord.x,
                            "y": coord.y,
                            "z": coord.z + dims.height,
                            "width": dims.width,
                            "length": dims.length,
                            "support_score": dims.width * dims.length,
                        }
                    )

    # Add ground surface if needed
    if not stacking_surfaces:
        stacking_surfaces.append(
            {
                "x": 0,
                "y": 0,
                "z": 0,
                "width": pallet_dims.width,
                "length": pallet_dims.length,
                "support_score": pallet_dims.width * pallet_dims.length,
            }
        )

    # Create population using different strategies
    for strategy_idx in range(population_size):
        try:
            # Create copies of the items to avoid modifying originals
            items_to_place = deepcopy(not_covered_superitems)

            # Sort based on strategy
            sort_strategy = strategy_idx % 5
            if sort_strategy == 0:
                # Sort by volume
                try:
                    items_to_place.sort(key=lambda x: getattr(x, "volume", 0), reverse=True)
                except:
                    # Fallback sorting
                    random.shuffle(items_to_place)
            elif sort_strategy == 1:
                # Sort by base area
                try:
                    items_to_place.sort(key=lambda x: safe_get_area(x), reverse=True)
                except:
                    random.shuffle(items_to_place)
            # Other sort strategies...
            else:
                # Random order
                random.shuffle(items_to_place)

            # Create chromosome with layers
            chromosome = []
            current_z = 0  # <-- Fix: Always start GA layers at z=0

            # Create layers until all items are placed or can't place more
            while items_to_place:
                layer_items = []
                layer_coords = []
                max_height = 0
                used_positions = set()

                # Try to place items in this layer
                i = 0
                items_placed = False

                while i < len(items_to_place):
                    item = items_to_place[i]

                    # Safe check dimensions
                    try:
                        item_width = safe_get_width(item)
                        item_length = safe_get_length(item)
                        item_height = safe_get_height(item)
                    except Exception as e:
                        # Skip problematic items
                        logger.warning(f"Error getting item dimensions: {e}")
                        i += 1
                        continue

                    # Simple bottom-left placement without complex rotation
                    # to avoid property access errors
                    placed = False

                    # Try placement on a grid
                    step = max(5, min(pallet_dims.width, pallet_dims.length) // 20)
                    for x in range(0, int(pallet_dims.width - item_width) + 1, step):
                        if placed:
                            break
                        for y in range(0, int(pallet_dims.length - item_length) + 1, step):
                            # Check bounds
                            if current_z + item_height > pallet_dims.height:
                                continue

                            # Check overlaps
                            overlap = False
                            for px, py, px2, py2 in used_positions:
                                if (
                                    x < px2
                                    and x + item_width > px
                                    and y < py2
                                    and y + item_length > py
                                ):
                                    overlap = True
                                    break

                            if not overlap:
                                # Place item
                                layer_items.append(item)
                                layer_coords.append(utils.Coordinate(x, y, current_z))
                                used_positions.add((x, y, x + item_width, y + item_length))
                                max_height = max(max_height, item_height)

                                # Remove from remaining items
                                items_to_place.pop(i)
                                placed = True
                                items_placed = True
                                break

                    if not placed:
                        i += 1

                # Add layer if items were placed
                if layer_items:
                    chromosome.append(
                        {
                            "items": layer_items,
                            "coords": layer_coords,
                            "height": max_height,
                            "z_level": current_z,
                        }
                    )
                    # Update z for next layer
                    current_z += max_height
                else:
                    # No more items can be placed in a new layer
                    break

            # Add chromosome to population
            if chromosome:
                population.append(chromosome)

        except Exception as e:
            logger.warning(f"Error generating chromosome: {str(e)}")
            continue

    # Fill up population if needed
    while len(population) < population_size and population:
        # Clone and modify existing chromosomes
        clone_idx = random.randint(0, len(population) - 1)
        try:
            clone = deepcopy(population[clone_idx])
            population.append(clone)
        except:
            # Just duplicate without modification if deepcopy fails
            population.append(population[clone_idx])

    # Return population (or empty if no valid chromosomes)
    return population


# Helper functions to safely get properties and avoid recursion errors
def safe_get_width(item):
    """Safely extract width from item with fallback handling."""
    if hasattr(item, "_width"):
        return item._width
    if hasattr(item, "width"):
        # Check if width is a property or an attribute
        if callable(getattr(item.__class__, "width", None)):
            # It's a property, call it directly
            try:
                return item.width
            except:
                # If error, try alternate ways
                if hasattr(item, "get_width"):
                    return item.get_width()
                if hasattr(item, "dimensions") and hasattr(item.dimensions, "width"):
                    return item.dimensions.width
        else:
            # It's an attribute
            return item.width
    # Default fallback
    return 0


def safe_get_length(item):
    """Safely extract length from item with fallback handling."""
    if hasattr(item, "_length"):
        return item._length
    if hasattr(item, "length"):
        # Check if length is a property or an attribute
        if callable(getattr(item.__class__, "length", None)):
            # It's a property, call it directly
            try:
                return item.length
            except:
                # If error, try alternate ways
                if hasattr(item, "get_length"):
                    return item.get_length()
                if hasattr(item, "dimensions") and hasattr(item.dimensions, "length"):
                    return item.dimensions.length
        else:
            # It's an attribute
            return item.length
    # Default fallback
    return 0


def safe_get_height(item):
    """Safely extract height from item with fallback handling."""
    if hasattr(item, "_height"):
        return item._height
    if hasattr(item, "height"):
        # Check if height is a property or an attribute
        if callable(getattr(item.__class__, "height", None)):
            # It's a property, call it directly
            try:
                return item.height
            except:
                # If error, try alternate ways
                if hasattr(item, "get_height"):
                    return item.get_height()
                if hasattr(item, "dimensions") and hasattr(item.dimensions, "height"):
                    return item.dimensions.height
        else:
            # It's an attribute
            return item.height
    # Default fallback
    return 0


def safe_get_area(item):
    """Safely calculate item base area with fallback handling."""
    try:
        return safe_get_width(item) * safe_get_length(item)
    except:
        return 0


def create_simple_chromosome(items, pallet_dims):
    """Create a simple chromosome with basic layer-by-layer placement."""
    """Create a simple chromosome as a fallback"""
    sorted_items = sorted(items, key=lambda x: x.width * x.length, reverse=True)
    chromosome = []
    z_level = 0

    while sorted_items:
        layer_items = []
        layer_coords = []
        max_height = 0

        used_positions = set()

        i = 0
        while i < len(sorted_items):
            item = sorted_items[i]

            # Try to place at a simple grid position
            placed = False
            for x in range(0, pallet_dims.width - item.width + 1, 10):
                for y in range(0, pallet_dims.length - item.length + 1, 10):
                    # Check if position is free
                    overlaps = False
                    for px, py, px2, py2 in used_positions:
                        if x < px2 and x + item.width > px and y < py2 and y + item.length > py:
                            overlaps = True
                            break

                    if not overlaps:
                        layer_items.append(item)
                        layer_coords.append(utils.Coordinate(x, y, z_level))
                        used_positions.add((x, y, x + item.width, y + item.length))
                        max_height = max(max_height, item.height)
                        placed = True
                        break

                if placed:
                    break

            if placed:
                sorted_items.pop(i)
            else:
                i += 1

        if layer_items:
            chromosome.append(
                {
                    "items": layer_items,
                    "coords": layer_coords,
                    "height": max_height,
                    "z_level": z_level,
                }
            )
            z_level += max_height
        else:
            break

    return chromosome


def generate_extreme_points(used_positions, pallet_dims):
    """Generate extreme points for item placement optimization."""
    """
    Generate a list of extreme points for item placement.
    These are the corners where an item can potentially be placed.
    
    Args:
        used_positions: Set of (x1, y1, x2, y2) tuples representing occupied spaces
        pallet_dims: Pallet dimensions
    
    Returns:
        List of (x, y) coordinates for potential placement
    """
    # Start with the origin
    extreme_points = [(0, 0)]

    # Add corner points from all used positions
    for x1, y1, x2, y2 in used_positions:
        # Add the 3 corner points (top-left, top-right, bottom-right)
        extreme_points.extend(
            [
                (x1, y2),  # Bottom-right of current item becomes top-left of new
                (x2, y1),  # Top-right
                (x2, y2),  # Bottom-right
            ]
        )

    # Remove duplicates and sort by distance from origin
    extreme_points = list(set(extreme_points))
    extreme_points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)

    # Filter points that are within the pallet bounds
    valid_points = []
    for x, y in extreme_points:
        if 0 <= x < pallet_dims.width and 0 <= y < pallet_dims.length:
            valid_points.append((x, y))

    return valid_points


def can_place_item(item, x, y, z, used_positions, pallet_dims):
    """Check if an item can be placed at the specified position."""
    """
    Check if an item can be placed at the given position without overlapping
    and within pallet bounds.
    
    Args:
        item: The superitem to place
        x, y, z: The coordinates to place the item
        used_positions: Set of (x1, y1, x2, y2) tuples representing occupied spaces
        pallet_dims: Pallet dimensions
    
    Returns:
        Boolean indicating if placement is valid
    """
    # Check pallet boundaries
    if (
        x + item.width > pallet_dims.width
        or y + item.length > pallet_dims.length
        or z + item.height > pallet_dims.height
    ):
        return False

    # Check for overlaps with existing items
    for used_x1, used_y1, used_x2, used_y2 in used_positions:
        if x < used_x2 and x + item.width > used_x1 and y < used_y2 and y + item.length > used_y1:
            return False

    # All checks passed
    return True


def crossover(parent1, parent2):
    """Perform layer-based crossover between parent chromosomes."""
    if not parent1 or not parent2:
        return parent1, parent2

    try:
        # Randomly select crossover point
        max_layers = min(len(parent1), len(parent2))
        if max_layers <= 1:
            return parent1, parent2

        crossover_point = random.randint(1, max_layers - 1)

        # Create offspring
        offspring1 = parent1[:crossover_point].copy()
        offspring2 = parent2[:crossover_point].copy()

        # Track items already in each offspring
        items_in_offspring1 = set()
        items_in_offspring2 = set()

        # Gather IDs of items already in the first part of each offspring
        for layer in offspring1:
            for item in layer["items"]:
                if isinstance(item.id, list):
                    items_in_offspring1.update(item.id)
                else:
                    items_in_offspring1.add(item.id)

        for layer in offspring2:
            for item in layer["items"]:
                if isinstance(item.id, list):
                    items_in_offspring2.update(item.id)
                else:
                    items_in_offspring2.add(item.id)

        # Add layers from second parent, filtering out duplicates
        for layer in parent2[crossover_point:]:
            new_layer = {
                "items": [],
                "coords": [],
                "height": layer["height"],
                "z_level": 0,  # Will adjust later
            }

            for i, item in enumerate(layer["items"]):
                # Check if item is already in offspring1
                item_ids = item.id if isinstance(item.id, list) else [item.id]
                if not any(id in items_in_offspring1 for id in item_ids):
                    new_layer["items"].append(item)
                    new_layer["coords"].append(layer["coords"][i])
                    items_in_offspring1.update(item_ids)

            # Only add non-empty layers
            if new_layer["items"]:
                # If no items were kept, recompute height
                if len(new_layer["items"]) < len(layer["items"]):
                    new_layer["height"] = max(item.height for item in new_layer["items"])
                offspring1.append(new_layer)

        # Repeat for the second offspring
        for layer in parent1[crossover_point:]:
            new_layer = {
                "items": [],
                "coords": [],
                "height": layer["height"],
                "z_level": 0,  # Will adjust later
            }

            for i, item in enumerate(layer["items"]):
                item_ids = item.id if isinstance(item.id, list) else [item.id]
                if not any(id in items_in_offspring2 for id in item_ids):
                    new_layer["items"].append(item)
                    new_layer["coords"].append(layer["coords"][i])
                    items_in_offspring2.update(item_ids)

            if new_layer["items"]:
                if len(new_layer["items"]) < len(layer["items"]):
                    new_layer["height"] = max(item.height for item in new_layer["items"])
                offspring2.append(new_layer)

        # Adjust z_levels in both offspring
        for child in [offspring1, offspring2]:
            current_z = 0
            for i, layer in enumerate(child):
                layer["z_level"] = current_z
                current_z += layer["height"]

        return offspring1, offspring2
    except Exception as e:
        logger.warning(f"Error in crossover: {str(e)}")
        return parent1, parent2


def mutate(chromosome, not_covered_superitems, pallet_dims, mutation_rate=0.1):
    """Apply KPI-targeted mutation operations."""
    if not chromosome or random.random() > mutation_rate:
        return chromosome

    try:
        # Choose a mutation strategy aligned with KPIs
        mutation_type = random.choice(
            [
                "center_tall_items",  # Improve HeightWidthRatio KPI
                "fill_holes",  # Improve RelativeDensity KPI
                "compact_arrangement",  # Improve AbsoluteDensity KPI
                "optimize_contact",  # Improve SideSupport KPI
                "improve_support",  # Improve SurfaceSupport KPI
                "balance_center",  # Improve CenterOfGravity KPI
                "swap_items",  # General variation
                "swap_layers",  # General variation
                "add_remove_layer",  # General variation
            ]
        )

        mutated = deepcopy(chromosome)

        if mutation_type == "center_tall_items" and len(mutated) > 0:
            # Move tall items toward center and lower heights
            all_items = []
            all_coords = []
            for layer in mutated:
                all_items.extend(layer["items"])
                all_coords.extend(layer["coords"])

            # Find tall items
            tall_items = []
            for i, item in enumerate(all_items):
                if item.height > 0 and item.width > 0 and item.length > 0:
                    height_base_ratio = item.height / ((item.width * item.length) ** 0.5)
                    if height_base_ratio > 1.5:  # Significantly tall
                        tall_items.append((i, height_base_ratio))

            # Fix: Only proceed if we found tall items
            if tall_items:
                # Sort by height ratio (tallest first)
                tall_items.sort(key=lambda x: x[1], reverse=True)

                # Fix: Make sure we don't access an index out of range
                if len(tall_items) > 0:
                    # Try to move a random tall item
                    idx, _ = random.choice(tall_items[: min(3, len(tall_items))])
                    item = all_items[idx]
                    coord = all_coords[idx]

                    # Find which layer contains this item
                    for layer_idx, layer in enumerate(mutated):
                        if item in layer["items"]:
                            item_idx = layer["items"].index(item)

                            # Try to place closer to center
                            pallet_center_x = pallet_dims.width / 2
                            pallet_center_y = pallet_dims.length / 2

                            # Calculate position closer to center
                            new_x = max(
                                0,
                                min(
                                    pallet_dims.width - item.width, pallet_center_x - item.width / 2
                                ),
                            )
                            new_y = max(
                                0,
                                min(
                                    pallet_dims.length - item.length,
                                    pallet_center_y - item.length / 2,
                                ),
                            )

                            # Check for overlaps
                            overlap = False
                            for i, other_item in enumerate(layer["items"]):
                                if i != item_idx:
                                    other_coord = layer["coords"][i]
                                    if (
                                        new_x < other_coord.x + other_item.width
                                        and new_x + item.width > other_coord.x
                                        and new_y < other_coord.y + other_item.length
                                        and new_y + item.length > other_coord.y
                                    ):
                                        overlap = True
                                        break

                            if not overlap:
                                # Update coordinates
                                layer["coords"][item_idx] = utils.Coordinate(new_x, new_y, coord.z)
                            break

        elif mutation_type == "fill_holes" and len(mutated) > 0:
            # Fix: Only attempt if there are at least 2 layers
            if len(mutated) >= 2:
                # Select two random adjacent layers
                layer_idx = random.randint(0, len(mutated) - 2)
                lower_layer = mutated[layer_idx]
                upper_layer = mutated[layer_idx + 1]

                # Fix: Only proceed if both layers have items
                if upper_layer["items"] and lower_layer["items"]:
                    # Sort upper items by size (smallest first)
                    upper_items = [
                        (i, item.width * item.length * item.height)
                        for i, item in enumerate(upper_layer["items"])
                    ]
                    upper_items.sort(key=lambda x: x[1])

                    # Try with a small item
                    if upper_items:
                        item_idx, _ = upper_items[0]
                        # Fix: Check if item_idx is valid
                        if 0 <= item_idx < len(upper_layer["items"]):
                            item = upper_layer["items"][item_idx]

                            # Look for a hole in lower layer
                            lower_used = set()
                            for i, lower_item in enumerate(lower_layer["items"]):
                                coord = lower_layer["coords"][i]
                                lower_used.add(
                                    (
                                        coord.x,
                                        coord.y,
                                        coord.x + lower_item.width,
                                        coord.y + lower_item.length,
                                    )
                                )

                            # Create a grid of potential positions
                            step = max(1, min(item.width, item.length) // 2)
                            if step <= 0:  # Fix: Ensure step is positive
                                step = 1

                            moved = False
                            for x in range(0, int(pallet_dims.width - item.width) + 1, step):
                                if moved:
                                    break
                                for y in range(0, int(pallet_dims.length - item.length) + 1, step):
                                    # Check if position is inside a hole
                                    inside_hole = True
                                    for lx, ly, lx2, ly2 in lower_used:
                                        if (
                                            x < lx2
                                            and x + item.width > lx
                                            and y < ly2
                                            and y + item.length > ly
                                        ):
                                            inside_hole = False
                                            break

                                    if inside_hole:
                                        # Move item to this position
                                        lower_layer["items"].append(item)
                                        lower_layer["coords"].append(
                                            utils.Coordinate(x, y, lower_layer["z_level"])
                                        )
                                        # Fix: Check if indices are still valid before removing
                                        if item_idx < len(upper_layer["items"]) and item_idx < len(
                                            upper_layer["coords"]
                                        ):
                                            upper_layer["items"].pop(item_idx)
                                            upper_layer["coords"].pop(item_idx)
                                            moved = True
                                            break

        elif mutation_type == "compact_arrangement" and len(mutated) > 0:
            # Choose a random layer to compact
            layer_idx = random.randint(0, len(mutated) - 1)
            layer = mutated[layer_idx]

            # Try to compact the layout by moving items towards the origin (0,0)
            if layer["items"]:
                # Sort items by distance from origin
                items_with_distance = []
                for i, item in enumerate(layer["items"]):
                    coord = layer["coords"][i]
                    distance = (coord.x**2 + coord.y**2) ** 0.5  # Distance from origin
                    items_with_distance.append((i, distance))

                # Sort by distance (furthest first)
                items_with_distance.sort(key=lambda x: x[1], reverse=True)

                # Try to move a random item closer to origin
                if items_with_distance:
                    # Pick a random item among the furthest ones
                    idx, _ = random.choice(items_with_distance[: min(3, len(items_with_distance))])
                    item = layer["items"][idx]
                    coord = layer["coords"][idx]

                    # Try to move closer to origin
                    step_size = max(1, min(item.width, item.length) // 4)
                    if step_size <= 0:  # Ensure step is positive
                        step_size = 1

                    # Calculate direction towards origin
                    dx = -1 if coord.x > 0 else 0
                    dy = -1 if coord.y > 0 else 0

                    # Skip items already at the origin
                    if dx != 0 or dy != 0:
                        # Try moving in the direction of the origin
                        new_x = max(0, coord.x + dx * step_size)
                        new_y = max(0, coord.y + dy * step_size)

                        # Check for overlaps
                        overlap = False
                        for i, other_item in enumerate(layer["items"]):
                            if i != idx:
                                other_coord = layer["coords"][i]
                                if (
                                    new_x < other_coord.x + other_item.width
                                    and new_x + item.width > other_coord.x
                                    and new_y < other_coord.y + other_item.length
                                    and new_y + item.length > other_coord.y
                                ):
                                    overlap = True
                                    break

                        if not overlap:
                            # Update coordinates
                            layer["coords"][idx] = utils.Coordinate(new_x, new_y, coord.z)

        elif mutation_type == "optimize_contact" and len(mutated) > 0:
            # Improve side support by moving items to touch each other
            if len(mutated) > 0:
                # Choose a random layer
                layer_idx = random.randint(0, len(mutated) - 1)
                layer = mutated[layer_idx]

                if len(layer["items"]) >= 2:  # Need at least 2 items to optimize contact
                    # Choose a random item
                    item_idx = random.randint(0, len(layer["items"]) - 1)
                    item = layer["items"][item_idx]
                    coord = layer["coords"][item_idx]

                    # Calculate current contact with other items
                    current_contacts = 0
                    for i, other_item in enumerate(layer["items"]):
                        if i != item_idx:
                            other_coord = layer["coords"][i]
                            # Check if items are touching on the sides
                            if (
                                coord.x + item.width == other_coord.x
                                or other_coord.x + other_item.width == coord.x
                            ) and (
                                coord.y < other_coord.y + other_item.length
                                and coord.y + item.length > other_coord.y
                            ):
                                current_contacts += 1
                            if (
                                coord.y + item.length == other_coord.y
                                or other_coord.y + other_item.length == coord.y
                            ) and (
                                coord.x < other_coord.x + other_item.width
                                and coord.x + item.width > other_coord.x
                            ):
                                current_contacts += 1

                    # Try to optimize by moving the item
                    best_contacts = current_contacts
                    best_pos = None

                    # Try moving in different directions
                    for dx, dy in [(5, 0), (-5, 0), (0, 5), (0, -5)]:
                        new_x = max(0, min(pallet_dims.width - item.width, coord.x + dx))
                        new_y = max(0, min(pallet_dims.length - item.length, coord.y + dy))

                        # Skip if no movement
                        if new_x == coord.x and new_y == coord.y:
                            continue

                        # Check for overlaps
                        overlap = False
                        for i, other_item in enumerate(layer["items"]):
                            if i != item_idx:
                                other_coord = layer["coords"][i]
                                if (
                                    new_x < other_coord.x + other_item.width
                                    and new_x + item.width > other_coord.x
                                    and new_y < other_coord.y + other_item.length
                                    and new_y + item.length > other_coord.y
                                ):
                                    overlap = True
                                    break

                        if not overlap:
                            # Calculate new contacts
                            new_contacts = 0
                            for i, other_item in enumerate(layer["items"]):
                                if i != item_idx:
                                    other_coord = layer["coords"][i]
                                    # Check if items would be touching on the sides
                                    if (
                                        new_x + item.width == other_coord.x
                                        or other_coord.x + other_item.width == new_x
                                    ) and (
                                        new_y < other_coord.y + other_item.length
                                        and new_y + item.length > other_coord.y
                                    ):
                                        new_contacts += 1
                                    if (
                                        new_y + item.length == other_coord.y
                                        or other_coord.y + other_item.length == new_y
                                    ) and (
                                        new_x < other_coord.x + other_item.width
                                        and new_x + item.width > other_coord.x
                                    ):
                                        new_contacts += 1

                            # Update if better
                            if new_contacts > best_contacts:
                                best_contacts = new_contacts
                                best_pos = (new_x, new_y)

                    # Apply the best position if found
                    if best_pos:
                        layer["coords"][item_idx] = utils.Coordinate(
                            best_pos[0], best_pos[1], coord.z
                        )

        elif mutation_type == "improve_support" and len(mutated) > 1:
            # Improve surface support by moving items to be better supported by the layer below
            # Only applicable if we have at least 2 layers

            # Choose a random layer (not the bottom one)
            layer_idx = random.randint(1, len(mutated) - 1)
            upper_layer = mutated[layer_idx]
            lower_layer = mutated[layer_idx - 1]

            if upper_layer["items"] and lower_layer["items"]:
                # Choose a random item from the upper layer
                item_idx = random.randint(0, len(upper_layer["items"]) - 1)
                item = upper_layer["items"][item_idx]
                coord = upper_layer["coords"][item_idx]

                # Calculate current support
                current_support = 0
                total_area = item.width * item.length

                for i, lower_item in enumerate(lower_layer["items"]):
                    lower_coord = lower_layer["coords"][i]

                    # Calculate overlap area
                    overlap_width = max(
                        0,
                        min(coord.x + item.width, lower_coord.x + lower_item.width)
                        - max(coord.x, lower_coord.x),
                    )
                    overlap_length = max(
                        0,
                        min(coord.y + item.length, lower_coord.y + lower_item.length)
                        - max(coord.y, lower_coord.y),
                    )

                    current_support += overlap_width * overlap_length

                current_support_ratio = current_support / total_area if total_area > 0 else 0

                # Try to improve support by moving the item
                best_support = current_support_ratio
                best_pos = None

                # Try moving in different directions
                step = max(1, min(item.width, item.length) // 4)
                if step <= 0:  # Ensure step is positive
                    step = 1

                for dx in range(-20, 21, step):
                    for dy in range(-20, 21, step):
                        new_x = max(0, min(pallet_dims.width - item.width, coord.x + dx))
                        new_y = max(0, min(pallet_dims.length - item.length, coord.y + dy))

                        # Skip if no movement
                        if new_x == coord.x and new_y == coord.y:
                            continue

                        # Check for overlaps with other items in the same layer
                        overlap = False
                        for i, other_item in enumerate(upper_layer["items"]):
                            if i != item_idx:
                                other_coord = upper_layer["coords"][i]
                                if (
                                    new_x < other_coord.x + other_item.width
                                    and new_x + item.width > other_coord.x
                                    and new_y < other_coord.y + other_item.length
                                    and new_y + item.length > other_coord.y
                                ):
                                    overlap = True
                                    break

                        if not overlap:
                            # Calculate new support
                            new_support = 0

                            for i, lower_item in enumerate(lower_layer["items"]):
                                lower_coord = lower_layer["coords"][i]

                                # Calculate overlap area
                                overlap_width = max(
                                    0,
                                    min(new_x + item.width, lower_coord.x + lower_item.width)
                                    - max(new_x, lower_coord.x),
                                )
                                overlap_length = max(
                                    0,
                                    min(new_y + item.length, lower_coord.y + lower_item.length)
                                    - max(new_y, lower_coord.y),
                                )

                                new_support += overlap_width * overlap_length

                            new_support_ratio = new_support / total_area if total_area > 0 else 0

                            # Update if better
                            if new_support_ratio > best_support:
                                best_support = new_support_ratio
                                best_pos = (new_x, new_y)

                # Apply the best position if found
                if best_pos:
                    upper_layer["coords"][item_idx] = utils.Coordinate(
                        best_pos[0], best_pos[1], coord.z
                    )

        elif mutation_type == "balance_center" and len(mutated) > 0:
            # Improve center of gravity by balancing the weight distribution

            # Calculate current center of gravity
            total_weight = 0
            weighted_x = 0
            weighted_y = 0

            for layer in mutated:
                for i, item in enumerate(layer["items"]):
                    weight = (
                        item.width * item.length * item.height
                    )  # Use volume as a proxy for weight
                    coord = layer["coords"][i]
                    center_x = coord.x + item.width / 2
                    center_y = coord.y + item.length / 2

                    weighted_x += center_x * weight
                    weighted_y += center_y * weight
                    total_weight += weight

            if total_weight > 0:
                current_cog_x = weighted_x / total_weight
                current_cog_y = weighted_y / total_weight

                # Pallet center
                pallet_center_x = pallet_dims.width / 2
                pallet_center_y = pallet_dims.length / 2

                # Calculate distance from center
                cog_distance = (
                    (current_cog_x - pallet_center_x) ** 2 + (current_cog_y - pallet_center_y) ** 2
                ) ** 0.5

                # Choose a random layer
                layer_idx = random.randint(0, len(mutated) - 1)
                layer = mutated[layer_idx]

                if layer["items"]:
                    # Try to move a random item to improve center of gravity
                    item_idx = random.randint(0, len(layer["items"]) - 1)
                    item = layer["items"][item_idx]
                    coord = layer["coords"][item_idx]

                    # Try different positions
                    best_distance = cog_distance
                    best_pos = None

                    step = max(1, min(item.width, item.length) // 4)
                    if step <= 0:  # Ensure step is positive
                        step = 1

                    # Try positions that balance the COG
                    for dx in range(-20, 21, step):
                        for dy in range(-20, 21, step):
                            new_x = max(0, min(pallet_dims.width - item.width, coord.x + dx))
                            new_y = max(0, min(pallet_dims.length - item.length, coord.y + dy))

                            # Skip if no movement
                            if new_x == coord.x and new_y == coord.y:
                                continue

                            # Check for overlaps
                            overlap = False
                            for i, other_item in enumerate(layer["items"]):
                                if i != item_idx:
                                    other_coord = layer["coords"][i]
                                    if (
                                        new_x < other_coord.x + other_item.width
                                        and new_x + item.width > other_coord.x
                                        and new_y < other_coord.y + other_item.length
                                        and new_y + item.length > other_coord.y
                                    ):
                                        overlap = True
                                        break

                            if not overlap:
                                # Calculate new COG
                                weight = item.width * item.length * item.height
                                new_weighted_x = (
                                    weighted_x
                                    - (coord.x + item.width / 2) * weight
                                    + (new_x + item.width / 2) * weight
                                )
                                new_weighted_y = (
                                    weighted_y
                                    - (coord.y + item.length / 2) * weight
                                    + (new_y + item.length / 2) * weight
                                )

                                new_cog_x = new_weighted_x / total_weight
                                new_cog_y = new_weighted_y / total_weight

                                # Calculate new distance from center
                                new_distance = (
                                    (new_cog_x - pallet_center_x) ** 2
                                    + (new_cog_y - pallet_center_y) ** 2
                                ) ** 0.5

                                # Update if better
                                if new_distance < best_distance:
                                    best_distance = new_distance
                                    best_pos = (new_x, new_y)

                    # Apply the best position if found
                    if best_pos:
                        layer["coords"][item_idx] = utils.Coordinate(
                            best_pos[0], best_pos[1], coord.z
                        )

        elif mutation_type == "swap_items" and len(mutated) > 0:
            # Swap the positions of two randomly selected items
            if sum(len(layer["items"]) for layer in mutated) >= 2:  # Need at least 2 items
                # Choose source and destination layers
                if len(mutated) == 1:
                    src_layer_idx = dst_layer_idx = 0
                else:
                    src_layer_idx = random.randint(0, len(mutated) - 1)
                    dst_layer_idx = random.randint(0, len(mutated) - 1)

                src_layer = mutated[src_layer_idx]
                dst_layer = mutated[dst_layer_idx]

                # Fix: Only proceed if both layers have items
                if src_layer["items"] and dst_layer["items"]:
                    # Choose random items
                    src_item_idx = random.randint(0, len(src_layer["items"]) - 1)
                    dst_item_idx = random.randint(0, len(dst_layer["items"]) - 1)

                    src_item = src_layer["items"][src_item_idx]
                    dst_item = dst_layer["items"][dst_item_idx]
                    src_coord = src_layer["coords"][src_item_idx]
                    dst_coord = dst_layer["coords"][dst_item_idx]

                    # Check if swap is feasible (items fit in each other's positions)
                    if (
                        src_item.width <= pallet_dims.width - dst_coord.x
                        and src_item.length <= pallet_dims.length - dst_coord.y
                        and dst_item.width <= pallet_dims.width - src_coord.x
                        and dst_item.length <= pallet_dims.length - src_coord.y
                    ):

                        # Check for overlaps after swap
                        src_overlap = False
                        dst_overlap = False

                        # Check source item in destination position
                        for i, other_item in enumerate(dst_layer["items"]):
                            if i != dst_item_idx:
                                other_coord = dst_layer["coords"][i]
                                if (
                                    dst_coord.x < other_coord.x + other_item.width
                                    and dst_coord.x + src_item.width > other_coord.x
                                    and dst_coord.y < other_coord.y + other_item.length
                                    and dst_coord.y + src_item.length > other_coord.y
                                ):
                                    src_overlap = True
                                    break

                        # Check destination item in source position
                        for i, other_item in enumerate(src_layer["items"]):
                            if i != src_item_idx:
                                other_coord = src_layer["coords"][i]
                                if (
                                    src_coord.x < other_coord.x + other_item.width
                                    and src_coord.x + dst_item.width > other_coord.x
                                    and src_coord.y < other_coord.y + other_item.length
                                    and src_coord.y + dst_item.length > other_coord.y
                                ):
                                    dst_overlap = True
                                    break

                        # Perform swap if no overlaps
                        if not (src_overlap or dst_overlap):
                            if src_layer_idx == dst_layer_idx:
                                # Swap within the same layer
                                (
                                    src_layer["items"][src_item_idx],
                                    src_layer["items"][dst_item_idx],
                                ) = (
                                    src_layer["items"][dst_item_idx],
                                    src_layer["items"][src_item_idx],
                                )

                                # Adjust coordinates for size differences
                                src_layer["coords"][src_item_idx] = utils.Coordinate(
                                    src_coord.x, src_coord.y, src_coord.z
                                )
                                src_layer["coords"][dst_item_idx] = utils.Coordinate(
                                    dst_coord.x, dst_coord.y, dst_coord.z
                                )
                            else:
                                # Swap between different layers
                                src_z = src_layer["z_level"]
                                dst_z = dst_layer["z_level"]

                                # Update height if needed
                                if src_item.height > dst_layer["height"]:
                                    # Need to adjust all layers above dst_layer_idx
                                    height_diff = src_item.height - dst_layer["height"]
                                    dst_layer["height"] = src_item.height

                                    # Update z-levels for layers above
                                    for i in range(dst_layer_idx + 1, len(mutated)):
                                        mutated[i]["z_level"] += height_diff

                                if dst_item.height > src_layer["height"]:
                                    # Need to adjust all layers above src_layer_idx
                                    height_diff = dst_item.height - src_layer["height"]
                                    src_layer["height"] = dst_item.height

                                    # Update z-levels for layers above
                                    for i in range(src_layer_idx + 1, len(mutated)):
                                        mutated[i]["z_level"] += height_diff

                                # Perform the swap
                                src_layer["items"][src_item_idx] = dst_item
                                dst_layer["items"][dst_item_idx] = src_item
                                src_layer["coords"][src_item_idx] = utils.Coordinate(
                                    src_coord.x, src_coord.y, src_z
                                )
                                dst_layer["coords"][dst_item_idx] = utils.Coordinate(
                                    dst_coord.x, dst_coord.y, dst_z
                                )

        elif mutation_type == "swap_layers" and len(mutated) >= 2:
            # Swap two layers
            layer1_idx = random.randint(0, len(mutated) - 1)
            layer2_idx = random.randint(0, len(mutated) - 1)

            # Ensure we select different layers
            while layer2_idx == layer1_idx:
                if len(mutated) > 1:  # Only retry if there are multiple layers
                    layer2_idx = random.randint(0, len(mutated) - 1)
                else:
                    break  # No point continuing if there's only one layer

            if layer1_idx != layer2_idx:
                # Store z-levels and heights
                z1 = mutated[layer1_idx]["z_level"]
                h1 = mutated[layer1_idx]["height"]
                z2 = mutated[layer2_idx]["z_level"]
                h2 = mutated[layer2_idx]["height"]

                # Swap the layers' positions
                mutated[layer1_idx], mutated[layer2_idx] = mutated[layer2_idx], mutated[layer1_idx]

                # Adjust z-levels for swapped layers
                mutated[layer1_idx]["z_level"] = z1
                mutated[layer2_idx]["z_level"] = z2

                # Adjust coordinates for all items in the layers
                for i, coord in enumerate(mutated[layer1_idx]["coords"]):
                    mutated[layer1_idx]["coords"][i] = utils.Coordinate(coord.x, coord.y, z1)

                for i, coord in enumerate(mutated[layer2_idx]["coords"]):
                    mutated[layer2_idx]["coords"][i] = utils.Coordinate(coord.x, coord.y, z2)

                # Recompute all z-levels to ensure consistency
                mutated.sort(key=lambda x: x["z_level"])
                z_level = 0
                for i, layer in enumerate(mutated):
                    layer["z_level"] = z_level
                    z_level += layer["height"]

                    # Update all item coordinates in this layer
                    for j, coord in enumerate(layer["coords"]):
                        layer["coords"][j] = utils.Coordinate(coord.x, coord.y, layer["z_level"])

        elif mutation_type == "add_remove_layer":
            # Modified add/remove layer with fixes for the pop index issue
            if random.random() < 0.5 and len(mutated) > 1:
                # Remove a random layer and redistribute items
                remove_idx = random.randint(0, len(mutated) - 1)
                items_to_redistribute = mutated[remove_idx][
                    "items"
                ].copy()  # Fix: Make a copy to avoid modifying while iterating

                # Remove the layer
                removed_z = mutated[remove_idx]["z_level"]
                removed_height = mutated[remove_idx]["height"]
                del mutated[remove_idx]

                # Adjust z-levels
                for i in range(len(mutated)):
                    if mutated[i]["z_level"] > removed_z:
                        mutated[i]["z_level"] -= removed_height

                # Try to redistribute items to other layers
                for item in items_to_redistribute:
                    placed = False

                    # Try each layer
                    for layer_idx, layer in enumerate(mutated):
                        z_level = layer["z_level"]

                        # Try to find a valid position
                        for x in range(0, pallet_dims.width - item.width + 1, 5):
                            if placed:
                                break
                            for y in range(0, pallet_dims.length - item.length + 1, 5):
                                # Check for overlaps
                                overlap = False
                                for i, other_item in enumerate(layer["items"]):
                                    other_coord = layer["coords"][i]
                                    if (
                                        x < other_coord.x + other_item.width
                                        and x + item.width > other_coord.x
                                        and y < other_coord.y + other_item.length
                                        and y + item.length > other_coord.y
                                    ):
                                        overlap = True
                                        break

                                if not overlap:
                                    # Place item
                                    layer["items"].append(item)
                                    layer["coords"].append(utils.Coordinate(x, y, z_level))

                                    # Update layer height if needed
                                    if item.height > layer["height"]:
                                        # Adjust subsequent layers
                                        height_diff = item.height - layer["height"]
                                        layer["height"] = item.height

                                        # Update z-levels for layers above
                                        for i in range(layer_idx + 1, len(mutated)):
                                            mutated[i]["z_level"] += height_diff

                                    placed = True
                                    break
            else:
                # Add a new layer with unused items
                flat_items_ids = set()
                for layer in mutated:
                    for item in layer["items"]:
                        if isinstance(item.id, list):
                            flat_items_ids.update(item.id)
                        else:
                            flat_items_ids.add(item.id)

                # Find unused items
                unused_items = []
                for item in not_covered_superitems:
                    item_ids = item.id if isinstance(item.id, list) else [item.id]
                    if not any(id in flat_items_ids for id in item_ids):
                        unused_items.append(item)

                # Fix: Only proceed if there are unused items
                if unused_items:
                    # Get the current top z-level
                    z_level = 0
                    if mutated:
                        last_layer = mutated[-1]
                        z_level = last_layer["z_level"] + last_layer["height"]

                    # Create a new layer with compact packing
                    layer_items = []
                    layer_coords = []
                    used_positions = set()
                    max_height = 0

                    # Sort by area for better packing
                    unused_items.sort(key=lambda x: x.width * x.length, reverse=True)

                    # Fix: Make sure we don't try to access more items than available
                    items_to_try = unused_items[: min(5, len(unused_items))]

                    # Add up to 5 items to the new layer
                    for item in items_to_try:
                        # Check if item fits
                        if z_level + item.height <= pallet_dims.height:
                            placed = False

                            # Try to find a valid position
                            for x in range(0, pallet_dims.width - item.width + 1, 5):
                                if placed:
                                    break
                                for y in range(0, pallet_dims.length - item.length + 1, 5):
                                    # Check for overlaps
                                    overlap = False
                                    for px, py, px2, py2 in used_positions:
                                        if (
                                            x < px2
                                            and x + item.width > px
                                            and y < py2
                                            and y + item.length > py
                                        ):
                                            overlap = True
                                            break

                                    if not overlap:
                                        layer_items.append(item)
                                        layer_coords.append(utils.Coordinate(x, y, z_level))
                                        used_positions.add((x, y, x + item.width, y + item.length))
                                        max_height = max(max_height, item.height)
                                        placed = True
                                        break

                    # Add layer if items were placed
                    if layer_items:
                        mutated.append(
                            {
                                "items": layer_items,
                                "coords": layer_coords,
                                "height": max_height,
                                "z_level": z_level,
                            }
                        )

        # Clean up and recompute z-levels
        # Remove empty layers
        mutated = [layer for layer in mutated if layer["items"]]

        # Fix z-levels
        if mutated:
            mutated[0]["z_level"] = 0
            for i in range(1, len(mutated)):
                mutated[i]["z_level"] = mutated[i - 1]["z_level"] + mutated[i - 1]["height"]

        return mutated

    except Exception as e:
        logger.warning(f"Error in mutation: {str(e)}")
        return chromosome


def convert_to_layer_pool(chromosome, pallet_dims, base_z_level=0):
    """Convert a GA chromosome to a LayerPool for integration with the main system."""
    """
    Convert a chromosome to a LayerPool with coordinates already assigned,
    taking into account the base_z_level from the first phase.
    """
    layer_pool = layers.LayerPool(SuperitemPool(), pallet_dims)
    accumulated_z = base_z_level  # Initialize with the base_z_level

    # Handle empty chromosome
    if not chromosome:
        logger.warning("Empty chromosome in convert_to_layer_pool")
        return layer_pool

    try:
        # Process each layer in the chromosome
        for layer_data in chromosome:
            if not layer_data.get("items") or not layer_data.get("coords"):
                logger.warning(f"Invalid layer data: {layer_data}")
                continue

            # Verify items and coords have matching lengths
            if len(layer_data["items"]) != len(layer_data["coords"]):
                logger.warning(
                    f"Mismatched items and coords lengths: {len(layer_data['items'])} items vs {len(layer_data['coords'])} coords"
                )
                # Adjust to match
                min_len = min(len(layer_data["items"]), len(layer_data["coords"]))
                layer_data["items"] = layer_data["items"][:min_len]
                layer_data["coords"] = layer_data["coords"][:min_len]

            # Create SuperitemPool for this layer
            superitem_pool = SuperitemPool(superitems=layer_data["items"])

            # Create adjusted coordinates with the appropriate z-level offset
            adjusted_coords = []
            for coord in layer_data["coords"]:
                # Create a new coordinate with z adjusted by accumulated_z
                new_coord = utils.Coordinate(
                    coord.x,
                    coord.y,
                    accumulated_z + layer_data["z_level"],  # Add base_z_level via accumulated_z
                )
                adjusted_coords.append(new_coord)

            # Create Layer with adjusted coordinates
            layer = layers.Layer(superitem_pool, adjusted_coords, pallet_dims)

            # Add layer to pool
            layer_pool.add(layer)

            # Update accumulated_z for the next layer
            accumulated_z += layer_data["height"]

        return layer_pool
    except Exception as e:
        logger.error(f"Error in convert_to_layer_pool: {str(e)}")
        return layers.LayerPool(SuperitemPool(), pallet_dims)


def fitness_function(chromosome, target_superitems, pallet_dims, base_z_level=0):
    """Evaluate chromosome using multi-objective KPI metrics."""
    if not chromosome:
        return 0.0

    # Initialize KPI scores
    height_width_ratio_score = 0.0
    relative_density_score = 0.0
    absolute_density_score = 0.0
    side_support_score = 0.0
    surface_support_score = 0.0
    center_of_gravity_score = 0.0

    # Constraint violations
    overlap_penalty = 0.0

    try:
        # Get all packed items and their coordinates
        packed_items = []
        item_coords = []

        total_height = base_z_level
        for layer in chromosome:
            packed_items.extend(layer["items"])
            item_coords.extend(layer["coords"])
            total_height += layer["height"]

        # If no items are packed, return 0
        if not packed_items:
            return 0.0

        # 1. Calculate HeightWidthRatio KPI
        tall_item_penalty = 0.0
        total_items = len(packed_items)
        if total_items > 0:
            for i, item in enumerate(packed_items):
                coord = item_coords[i]

                # Calculate height-to-base ratio
                base_area = item.width * item.length
                if base_area > 0:
                    height_base_ratio = item.height / (base_area**0.5)

                    # Tall items (ratio > 1) should be more central and lower
                    if height_base_ratio > 1.0:
                        # Calculate distance from center
                        item_center_x = coord.x + item.width / 2
                        item_center_y = coord.y + item.length / 2
                        pallet_center_x = pallet_dims.width / 2
                        pallet_center_y = pallet_dims.length / 2

                        dist_from_center = (
                            (item_center_x - pallet_center_x) ** 2
                            + (item_center_y - pallet_center_y) ** 2
                        ) ** 0.5

                        # Normalize by pallet size
                        max_dist = (
                            (pallet_dims.width / 2) ** 2 + (pallet_dims.length / 2) ** 2
                        ) ** 0.5
                        normalized_dist = dist_from_center / max_dist if max_dist > 0 else 0

                        # Normalize height position
                        normalized_height = (
                            coord.z / pallet_dims.height if pallet_dims.height > 0 else 0
                        )

                        # Calculate penalty based on both distance from center and height
                        # Taller items with higher ratios get higher penalties
                        item_penalty = height_base_ratio * (
                            0.7 * normalized_dist + 0.3 * normalized_height
                        )
                        tall_item_penalty += item_penalty / total_items

            # Convert penalty to score (0-1)
            height_width_ratio_score = max(0.0, min(1.0, 1.0 - tall_item_penalty))

        # 2. Calculate RelativeDensity KPI (space utilization efficiency)
        # Identify the max z-level used (total height of packing)
        max_z = 0
        for coord in item_coords:
            idx = item_coords.index(coord)
            item_height = packed_items[idx].height
            max_z = max(max_z, coord.z + item_height)

        # Calculate total volume used
        used_volume = pallet_dims.width * pallet_dims.length * max_z

        # Count empty spaces (holes) below the upper envelope
        hole_count = 0
        filled_volume = 0

        # Simple grid-based approach
        grid_size = 20  # Resolution of the grid
        voxel_width = pallet_dims.width / grid_size
        voxel_length = pallet_dims.length / grid_size
        voxel_height = max_z / grid_size if max_z > 0 else 1

        # Create a 3D grid to track filled voxels
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

        # Fill grid with items
        for i, item in enumerate(packed_items):
            coord = item_coords[i]
            x_start = int(coord.x / voxel_width)
            y_start = int(coord.y / voxel_length)
            z_start = int(coord.z / voxel_height) if voxel_height > 0 else 0

            x_end = min(grid_size, int((coord.x + item.width) / voxel_width) + 1)
            y_end = min(grid_size, int((coord.y + item.length) / voxel_length) + 1)
            z_end = (
                min(grid_size, int((coord.z + item.height) / voxel_height) + 1)
                if voxel_height > 0
                else 1
            )

            # Mark voxels as filled
            grid[x_start:x_end, y_start:y_end, z_start:z_end] = True

        # Find upper envelope and count holes
        upper_envelope = np.zeros((grid_size, grid_size), dtype=int)
        for x in range(grid_size):
            for y in range(grid_size):
                occupied_z = np.where(grid[x, y, :])[0]
                if len(occupied_z) > 0:
                    upper_envelope[x, y] = occupied_z.max() + 1

        # Count holes and calculate utilized volume
        hole_count = 0
        utilized_volume = 0

        for x in range(grid_size):
            for y in range(grid_size):
                z_max = upper_envelope[x, y]
                if z_max > 0:
                    utilized_volume += z_max
                    # Count empty voxels below z_max
                    for z in range(z_max):
                        if not grid[x, y, z]:
                            hole_count += 1

        # Calculate density score
        if utilized_volume > 0:
            hole_ratio = hole_count / utilized_volume
            relative_density_score = 1.0 - min(1.0, hole_ratio)
        else:
            relative_density_score = 0.0

        # 3. Calculate AbsoluteDensity KPI (volume utilization)
        total_item_volume = sum(item.volume for item in packed_items)
        used_bin_volume = pallet_dims.width * pallet_dims.length * max_z

        if used_bin_volume > 0:
            absolute_density_score = min(1.0, total_item_volume / used_bin_volume)
        else:
            absolute_density_score = 0.0

        # 4. Calculate SideSupport KPI (side adjacency)
        total_sides = 0
        supported_sides = 0
        min_overlap_ratio = 0.2  # Minimum overlap to consider sides supported

        for i, item in enumerate(packed_items):
            coord = item_coords[i]

            # Define the 4 sides (excluding top and bottom)
            sides = [
                # Left side: (axis, position, range_start1, range_end1, range_start2, range_end2)
                ("x", coord.x, coord.y, coord.y + item.length, coord.z, coord.z + item.height),
                # Right side
                (
                    "x",
                    coord.x + item.width,
                    coord.y,
                    coord.y + item.length,
                    coord.z,
                    coord.z + item.height,
                ),
                # Front side
                ("y", coord.y, coord.x, coord.x + item.width, coord.z, coord.z + item.height),
                # Back side
                (
                    "y",
                    coord.y + item.length,
                    coord.x,
                    coord.x + item.width,
                    coord.z,
                    coord.z + item.height,
                ),
            ]

            # Check each side
            for side_axis, side_pos, start1, end1, start2, end2 in sides:
                # Skip sides at bin boundaries
                if (side_axis == "x" and (side_pos == 0 or side_pos == pallet_dims.width)) or (
                    side_axis == "y" and (side_pos == 0 or side_pos == pallet_dims.length)
                ):
                    continue

                # Calculate side area
                side_area = (end1 - start1) * (end2 - start2)
                total_sides += 1

                # Check if side has support from other items
                supported_area = 0.0

                for j, other_item in enumerate(packed_items):
                    if i == j:  # Skip self
                        continue

                    other_coord = item_coords[j]

                    # Check if adjacent
                    if side_axis == "x":
                        if side_pos == coord.x:  # Left side
                            if abs(other_coord.x + other_item.width - coord.x) < 0.1:
                                # Calculate overlap area
                                y_overlap = max(
                                    0,
                                    min(end1, other_coord.y + other_item.length)
                                    - max(start1, other_coord.y),
                                )
                                z_overlap = max(
                                    0,
                                    min(end2, other_coord.z + other_item.height)
                                    - max(start2, other_coord.z),
                                )
                                supported_area += y_overlap * z_overlap
                        else:  # Right side
                            if abs(other_coord.x - (coord.x + item.width)) < 0.1:
                                y_overlap = max(
                                    0,
                                    min(end1, other_coord.y + other_item.length)
                                    - max(start1, other_coord.y),
                                )
                                z_overlap = max(
                                    0,
                                    min(end2, other_coord.z + other_item.height)
                                    - max(start2, other_coord.z),
                                )
                                supported_area += y_overlap * z_overlap
                    else:  # y-axis
                        if side_pos == coord.y:  # Front side
                            if abs(other_coord.y + other_item.length - coord.y) < 0.1:
                                x_overlap = max(
                                    0,
                                    min(end1, other_coord.x + other_item.width)
                                    - max(start1, other_coord.x),
                                )
                                z_overlap = max(
                                    0,
                                    min(end2, other_coord.z + other_item.height)
                                    - max(start2, other_coord.z),
                                )
                                supported_area += x_overlap * z_overlap
                        else:  # Back side
                            if abs(other_coord.y - (coord.y + item.length)) < 0.1:
                                x_overlap = max(
                                    0,
                                    min(end1, other_coord.x + other_item.width)
                                    - max(start1, other_coord.x),
                                )
                                z_overlap = max(
                                    0,
                                    min(end2, other_coord.z + other_item.height)
                                    - max(start2, other_coord.z),
                                )
                                supported_area += x_overlap * z_overlap

                # Calculate support ratio
                support_ratio = supported_area / side_area if side_area > 0 else 0

                # Count as supported if enough overlap
                if support_ratio >= min_overlap_ratio:
                    supported_sides += 1

        # Calculate side support score
        if total_sides > 0:
            side_support_score = supported_sides / total_sides
        else:
            side_support_score = 0.0

        # 5. Calculate SurfaceSupport KPI (bottom support)
        total_items_needing_support = 0
        supported_items = 0

        for i, item in enumerate(packed_items):
            coord = item_coords[i]

            # Skip items on the ground
            if abs(coord.z) < 0.1:
                continue

            total_items_needing_support += 1

            # Calculate item's bottom area
            item_bottom_area = item.width * item.length
            supported_area = 0.0

            # Check support from items below
            for j, support_item in enumerate(packed_items):
                if i == j:
                    continue

                support_coord = item_coords[j]

                # Check if directly below
                if abs(support_coord.z + support_item.height - coord.z) < 0.1:
                    # Calculate overlap area
                    x_overlap = max(
                        0,
                        min(coord.x + item.width, support_coord.x + support_item.width)
                        - max(coord.x, support_coord.x),
                    )
                    y_overlap = max(
                        0,
                        min(coord.y + item.length, support_coord.y + support_item.length)
                        - max(coord.y, support_coord.y),
                    )

                    if x_overlap > 0 and y_overlap > 0:
                        supported_area += x_overlap * y_overlap

            # Calculate support ratio
            support_ratio = supported_area / item_bottom_area if item_bottom_area > 0 else 0

            # Consider sufficient support (>75%)
            if support_ratio >= 0.75:
                supported_items += 1

        # Calculate surface support score
        if total_items_needing_support > 0:
            surface_support_score = supported_items / total_items_needing_support
        else:
            surface_support_score = 1.0  # All items on ground = perfect

        # 6. Calculate CenterOfGravity KPI (balance)
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0

        for i, item in enumerate(packed_items):
            coord = item_coords[i]
            volume = item.width * item.length * item.height

            # Item center coordinates
            center_x = coord.x + item.width / 2
            center_y = coord.y + item.length / 2

            weighted_x += center_x * volume
            weighted_y += center_y * volume
            total_mass += volume

        if total_mass > 0:
            # Calculate center of gravity
            cog_x = weighted_x / total_mass
            cog_y = weighted_y / total_mass

            # Calculate distance from center of pallet
            pallet_center_x = pallet_dims.width / 2
            pallet_center_y = pallet_dims.length / 2

            distance = ((cog_x - pallet_center_x) ** 2 + (cog_y - pallet_center_y) ** 2) ** 0.5

            # Normalize (0 = perfect, 1 = worst)
            max_distance = ((pallet_dims.width / 2) ** 2 + (pallet_dims.length / 2) ** 2) ** 0.5

            if max_distance > 0:
                center_of_gravity_score = 1.0 - (distance / max_distance)
            else:
                center_of_gravity_score = 1.0
        else:
            center_of_gravity_score = 0.0

        # 7. Calculate coverage (how many target items are packed)
        packed_ids = set()
        for item in packed_items:
            if isinstance(item.id, list):
                packed_ids.update(item.id)
            else:
                packed_ids.add(item.id)

        target_ids = set()
        for item in target_superitems:
            if isinstance(item.id, list):
                target_ids.update(item.id)
            else:
                target_ids.add(item.id)

        if target_ids:
            coverage_score = len(packed_ids & target_ids) / len(target_ids)
        else:
            coverage_score = 0.0

        # 8. Check for overlaps
        for i, item1 in enumerate(packed_items):
            coord1 = item_coords[i]
            for j, item2 in enumerate(packed_items):
                if i != j:
                    coord2 = item_coords[j]

                    # Calculate overlap
                    x_overlap = max(
                        0,
                        min(coord1.x + item1.width, coord2.x + item2.width)
                        - max(coord1.x, coord2.x),
                    )
                    y_overlap = max(
                        0,
                        min(coord1.y + item1.length, coord2.y + item2.length)
                        - max(coord1.y, coord2.y),
                    )
                    z_overlap = max(
                        0,
                        min(coord1.z + item1.height, coord2.z + item2.height)
                        - max(coord1.z, coord2.z),
                    )

                    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
                        overlap_volume = x_overlap * y_overlap * z_overlap
                        overlap_penalty += (overlap_volume / pallet_dims.volume) * 10.0

        # Calculate weighted fitness score
        # Use weights similar to those in kpi_analysis.py BinPackingEvaluator
        final_score = (
            0.45 * coverage_score  # Most important: packing as many items as possible
            + 0.15 * absolute_density_score  # Volume utilization
            + 0.10 * relative_density_score  # Space efficiency (holes)
            + 0.10 * side_support_score  # Side contact
            + 0.10 * surface_support_score  # Bottom support
            + 0.05 * height_width_ratio_score  # Tall item stability
            + 0.05 * center_of_gravity_score  # Balance
            - min(1.0, overlap_penalty)  # Major penalty for overlaps
        )

        # Ensure fitness is non-negative
        return max(0.001, final_score)

    except Exception as e:
        logger.warning(f"Error in fitness calculation: {str(e)}")
        # Return minimal fitness
        return 0.001


def calculate_stability(chromosome):
    """Calculate structural stability score for a chromosome."""
    if not chromosome:
        return 0.0

    try:
        stability_score = 0.0
        total_items = 0

        # Start from the second layer (index 1)
        for i in range(1, len(chromosome)):
            layer = chromosome[i]
            prev_layer = chromosome[i - 1]

            # For each item in current layer
            for j, item in enumerate(layer["items"]):
                item_coord = layer["coords"][j]
                item_x = item_coord.x
                item_y = item_coord.y
                item_width = item.width
                item_length = item.length
                item_area = item_width * item_length

                # Calculate supported area
                supported_area = 0

                # Check support from items in previous layer
                for k, prev_item in enumerate(prev_layer["items"]):
                    prev_coord = prev_layer["coords"][k]
                    prev_x = prev_coord.x
                    prev_y = prev_coord.y
                    prev_width = prev_item.width
                    prev_length = prev_item.length

                    # Calculate overlap
                    x_overlap = max(
                        0, min(item_x + item_width, prev_x + prev_width) - max(item_x, prev_x)
                    )
                    y_overlap = max(
                        0, min(item_y + item_length, prev_y + prev_length) - max(item_y, prev_y)
                    )

                    # Add to supported area
                    supported_area += x_overlap * y_overlap

                # Calculate support ratio
                support_ratio = supported_area / item_area if item_area > 0 else 0

                # Add to stability score
                stability_score += support_ratio
                total_items += 1

        # Normalize stability score
        return stability_score / total_items if total_items > 0 else 1.0
    except Exception as e:
        logger.warning(f"Error calculating stability: {str(e)}")
        return 0.0


def calculate_compactness(chromosome):
    """Calculate compactness score based on inter-item contact areas."""
    if not chromosome:
        return 0.0

    total_contact_area = 0.0
    total_surface_area = 0.0

    for i, layer in enumerate(chromosome):
        for j, item in enumerate(layer["items"]):
            coord = layer["coords"][j]
            x1, y1, z1 = coord.x, coord.y, coord.z
            x2, y2, z2 = x1 + item.width, y1 + item.length, z1 + item.height

            # Calculate total surface area of this item
            item_surface_area = 2 * (
                item.width * item.length + item.width * item.height + item.length * item.height
            )
            total_surface_area += item_surface_area

            # Count contact with floor
            if z1 == 0:
                total_contact_area += item.width * item.length

            # Count contact with other items
            for other_i, other_layer in enumerate(chromosome):
                for other_j, other_item in enumerate(other_layer["items"]):
                    # Skip self
                    if i == other_i and j == other_j:
                        continue

                    other_coord = other_layer["coords"][other_j]
                    other_x1, other_y1, other_z1 = other_coord.x, other_coord.y, other_coord.z
                    other_x2 = other_x1 + other_item.width
                    other_y2 = other_y1 + other_item.length
                    other_z2 = other_z1 + other_item.height

                    # Horizontal adjacency in x direction (y and z overlap)
                    if (
                        (x1 == other_x2 or x2 == other_x1)
                        and max(y1, other_y1) < min(y2, other_y2)
                        and max(z1, other_z1) < min(z2, other_z2)
                    ):
                        y_overlap = min(y2, other_y2) - max(y1, other_y1)
                        z_overlap = min(z2, other_z2) - max(z1, other_z1)
                        total_contact_area += y_overlap * z_overlap

                    # Horizontal adjacency in y direction (x and z overlap)
                    if (
                        (y1 == other_y2 or y2 == other_y1)
                        and max(x1, other_x1) < min(x2, other_x2)
                        and max(z1, other_z1) < min(z2, other_z2)
                    ):
                        x_overlap = min(x2, other_x2) - max(x1, other_x1)
                        z_overlap = min(z2, other_z2) - max(z1, other_z1)
                        total_contact_area += x_overlap * z_overlap

                    # Vertical adjacency (x and y overlap, z adjacent)
                    if (
                        (z1 == other_z2 or z2 == other_z1)
                        and max(x1, other_x1) < min(x2, other_x2)
                        and max(y1, other_y1) < min(y2, other_y2)
                    ):
                        x_overlap = min(x2, other_x2) - max(x1, other_x1)
                        y_overlap = min(y2, other_y2) - max(y1, other_y1)
                        total_contact_area += x_overlap * y_overlap

    # Avoid division by zero
    if total_surface_area == 0:
        return 0.001

    # Calculate the compactness ratio (0.0 to 1.0)
    compactness = total_contact_area / total_surface_area
    return compactness
