import math

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.utils import utils
from src.utils.visualization import get_pallet_plot, plot_product


def visualize_chromosome(chromosome, pallet_dims, title="Chromosome Visualization"):
    """Visualize a single chromosome (solution) from GA."""
    if not chromosome:
        logger.warning("Empty chromosome, nothing to visualize")
        return None

    # Create the plot
    ax = get_pallet_plot(pallet_dims)

    # Update z-axis limit based on actual solution height
    max_z = (
        max(layer["z_level"] + layer["height"] for layer in chromosome)
        if chromosome
        else pallet_dims.height
    )
    ax.set_zlim([0, max(max_z * 1.1, pallet_dims.height)])  # Add 10% margin

    # Add title
    ax.set_title(title)

    # Disable grid lines and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Plot each layer in the chromosome
    for layer_idx, layer in enumerate(chromosome):
        for item_idx, item in enumerate(layer["items"]):
            coord = layer["coords"][item_idx]
            item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
            dims = utils.Dimension(item.width, item.length, item.height)
            plot_product(ax, item_id, coord, dims, pallet_dims, alpha=0.75)

    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the current figure


def visualize_population(population, pallet_dims, max_chromosomes=4, title="Population Example"):
    """Visualize a sample of chromosomes from the population."""
    if not population:
        logger.warning("Empty population, nothing to visualize")
        return None

    # Determine how many chromosomes to display
    num_to_display = min(max_chromosomes, len(population))

    # Choose chromosomes to display (random sample if too many)
    if len(population) > max_chromosomes:
        import random

        indices = random.sample(range(len(population)), num_to_display)
        display_chromosomes = [population[i] for i in indices]
    else:
        display_chromosomes = population[:num_to_display]
        indices = list(range(num_to_display))

    # Calculate grid utils.Dimensions
    cols = int(math.ceil(math.sqrt(num_to_display)))
    rows = int(math.ceil(num_to_display / cols))

    # Create figure
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 2), subplot_kw={"projection": "3d"}
    )

    # Handle single subplot case
    if num_to_display == 1:
        axes = np.array([axes])

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Calculate max height used across all chromosomes for consistent display
    all_max_z = 0
    for chromosome in display_chromosomes:
        if chromosome:
            max_z = max(layer["z_level"] + layer["height"] for layer in chromosome)
            all_max_z = max(all_max_z, max_z)

    # Start with full pallet height for plotting
    initial_z_limit = pallet_dims.height

    # Plot each chromosome
    for i, (ax, chromosome, idx) in enumerate(zip(axes, display_chromosomes, indices)):
        # Set axis limits and labels
        ax.set_xlim([0, pallet_dims.width])
        ax.set_ylim([0, pallet_dims.length])
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

        # Initially use full pallet height for plotting
        ax.set_zlim([0, initial_z_limit])

        ax.set_xlabel("Width", fontsize=8)
        ax.set_ylabel("Length", fontsize=8)
        ax.set_zlabel("Height", fontsize=8)

        # Disable grid lines and ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Calculate fitness metrics for title
        num_layers = len(chromosome)
        total_items = sum(len(layer["items"]) for layer in chromosome)

        ax.set_title(
            f"Chromosome {idx}\nLayers: {num_layers}, Items: {total_items}", fontsize=9, pad=5
        )

        # Plot each layer in the chromosome
        for layer in chromosome:
            for item_idx, item in enumerate(layer["items"]):
                coord = layer["coords"][item_idx]
                item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
                dims = utils.Dimension(item.width, item.length, item.height)
                plot_product(ax, item_id, coord, dims, pallet_dims, alpha=0.75)

    # Turn off any unused subplots
    for j in range(num_to_display, len(axes)):
        axes[j].axis("off")

    # After plotting, crop to max height used with small margin
    final_z_limit = all_max_z * 1.1 if all_max_z > 0 else pallet_dims.height
    for i in range(num_to_display):
        axes[i].set_zlim([0, final_z_limit])
        # Update box_aspect to match the cropped height
        axes[i].set_box_aspect([pallet_dims.width, pallet_dims.length, final_z_limit])

    # Add main title
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    return fig


def visualize_initialization_strategies(population, pallet_dims, title="Initialization Strategies"):
    """Visualize different initialization strategies from the GA population."""
    if not population:
        logger.warning("Empty population, nothing to visualize")
        return None

    # Collect examples from different parts of the population
    pop_size = len(population)

    if pop_size < 2:
        logger.warning("Population too small for strategy comparison")
        return None

    # Calculate statistics for each chromosome to find diverse examples
    chromosome_stats = []
    for idx, chrom in enumerate(population):
        # Calculate stats
        num_layers = len(chrom)
        total_items = sum(len(layer["items"]) for layer in chrom)
        total_volume = sum(
            sum(item.width * item.length * item.height for item in layer["items"])
            for layer in chrom
        )

        # Calculate height utilization
        max_height = max(layer["z_level"] + layer["height"] for layer in chrom) if chrom else 0
        height_utilization = max_height / pallet_dims.height if pallet_dims.height > 0 else 0

        # Store stats with index
        chromosome_stats.append(
            {
                "index": idx,
                "num_layers": num_layers,
                "total_items": total_items,
                "total_volume": total_volume,
                "height_utilization": height_utilization,
            }
        )

    # Sort by different metrics to find diverse examples
    # 1. Sort by number of layers (fewer layers)
    by_layers = sorted(chromosome_stats, key=lambda x: x["num_layers"])
    # 2. Sort by total items (more items)
    by_items = sorted(chromosome_stats, key=lambda x: x["total_items"], reverse=True)
    # 3. Sort by height utilization (we want one with good and one with poor utilization)
    by_height = sorted(chromosome_stats, key=lambda x: x["height_utilization"])

    # Select diverse examples
    example1_idx = by_layers[0]["index"]  # Fewest layers

    # For the second example, find one with different characteristics
    # Try to get a chromosome with many items but different from the first
    for stat in by_items:
        if stat["index"] != example1_idx:
            example2_idx = stat["index"]
            break
    else:
        # Fallback: just take a different chromosome
        example2_idx = (example1_idx + 1) % pop_size

    # Prepare examples with descriptive names
    examples = [
        (population[example1_idx], "Example Chromosome 1"),
        (population[example2_idx], "Example Chromosome 2"),
    ]

    # Create figure with 1x2 grid (2 examples)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "3d"})

    # Calculate max height used across all examples for consistent display
    all_max_z = 0
    for chromosome, _ in examples:
        if chromosome:
            max_z = max(layer["z_level"] + layer["height"] for layer in chromosome)
            all_max_z = max(all_max_z, max_z)

    # Start with full pallet height for plotting
    initial_z_limit = pallet_dims.height

    for i, (ax, (chromosome, strategy_name)) in enumerate(zip(axes, examples)):
        # Set axis limits and labels
        ax.set_xlim([0, pallet_dims.width])
        ax.set_ylim([0, pallet_dims.length])
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

        # Initially use full pallet height for plotting
        ax.set_zlim([0, initial_z_limit])

        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_zlabel("Height")

        # Disable grid lines and ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Add title with strategy and metrics
        num_layers = len(chromosome)
        total_items = sum(len(layer["items"]) for layer in chromosome)

        # Calculate volume statistics
        total_volume = sum(
            sum(item.width * item.length * item.height for item in layer["items"])
            for layer in chromosome
        )
        volume_utilization = (
            total_volume / pallet_dims.volume * 100 if pallet_dims.volume > 0 else 0
        )

        ax.set_title(
            f"{strategy_name}\nLayers: {num_layers}, Items: {total_items}\nVolume Util: {volume_utilization:.1f}%"
        )

        # Plot each layer in the chromosome
        for layer in chromosome:
            for item_idx, item in enumerate(layer["items"]):
                coord = layer["coords"][item_idx]
                item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
                dims = utils.Dimension(item.width, item.length, item.height)

                # Plot with color based on layer height
                plot_product(ax, item_id, coord, dims, pallet_dims, alpha=0.75)

    # After plotting, crop to max height used with small margin
    final_z_limit = all_max_z * 1.1 if all_max_z > 0 else pallet_dims.height
    for ax in axes:
        ax.set_zlim([0, final_z_limit])
        # Update box_aspect to match the cropped height
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, final_z_limit])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    return fig


def visualize_crossover(parent1, parent2, child1, child2, pallet_dims):
    """Visualize the crossover operation showing parents and children with highlighted changes."""
    # Determine actual crossover point by analyzing item patterns
    # This is more accurate than just taking the middle
    max_layers = min(len(parent1), len(parent2))
    if max_layers <= 1:
        crossover_point = 0
    else:
        # Try to identify where crossover occurred based on layer similarity
        # Compare child1 layers to parent1 and parent2 layers
        crossover_point = 1  # Default to first layer if can't detect
        for i in range(1, max_layers):
            # Check if child1's layer i is more similar to parent2's layer i
            # and child1's layer i-1 is more similar to parent1's layer i-1
            if i < len(child1) and i - 1 < len(child1):
                c1_layer_items = set(
                    item.id for item in child1[i]["items"] if not isinstance(item.id, list)
                )
                p1_layer_items = set(
                    item.id for item in parent1[i]["items"] if not isinstance(item.id, list)
                )
                p2_layer_items = set(
                    item.id for item in parent2[i]["items"] if not isinstance(item.id, list)
                )

                prev_c1_items = set(
                    item.id for item in child1[i - 1]["items"] if not isinstance(item.id, list)
                )
                prev_p1_items = set(
                    item.id for item in parent1[i - 1]["items"] if not isinstance(item.id, list)
                )

                # If this layer shows a switch in similarity, mark it as crossover point
                if (
                    len(c1_layer_items.intersection(p2_layer_items))
                    > len(c1_layer_items.intersection(p1_layer_items))
                    and len(prev_c1_items.intersection(prev_p1_items)) > 0
                ):
                    crossover_point = i
                    break

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()

    chromosomes = [parent1, parent2, child1, child2]
    titles = ["Parent 1", "Parent 2", "Child 1", "Child 2"]

    # Extract all item IDs from parents and children for comparison
    def extract_ids(chromosome):
        ids = set()
        for layer in chromosome:
            for item in layer["items"]:
                if isinstance(item.id, list):
                    ids.update(item.id)
                else:
                    ids.add(item.id)
        return ids

    parent1_ids = extract_ids(parent1)
    parent2_ids = extract_ids(parent2)
    child1_ids = extract_ids(child1)
    child2_ids = extract_ids(child2)

    # Identify items that were inherited from each parent
    child1_from_p1 = child1_ids.intersection(parent1_ids)
    child1_from_p2 = child1_ids.intersection(parent2_ids)
    child2_from_p1 = child2_ids.intersection(parent1_ids)
    child2_from_p2 = child2_ids.intersection(parent2_ids)

    # Calculate max height used across all chromosomes for consistent display
    all_max_z = 0
    for chromosome in chromosomes:
        if chromosome:
            max_z = max(layer["z_level"] + layer["height"] for layer in chromosome)
            all_max_z = max(all_max_z, max_z)

    # Start with full pallet height for plotting
    initial_z_limit = pallet_dims.height

    # Plot each chromosome
    for i, (ax, chromosome, title) in enumerate(zip(axes, chromosomes, titles)):
        # Set axis limits and labels
        ax.set_xlim([0, pallet_dims.width])
        ax.set_ylim([0, pallet_dims.length])
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

        # Initially use full pallet height for plotting
        ax.set_zlim([0, initial_z_limit])

        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_zlabel("Height")

        # Disable gridlines and ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Add title with chromosome metrics
        num_layers = len(chromosome)
        total_items = sum(len(layer["items"]) for layer in chromosome)
        ax.set_title(f"{title}\nLayers: {num_layers}, Items: {total_items}")

        # Plot each layer in the chromosome with color coding for inheritance
        for layer_idx, layer in enumerate(chromosome):
            # Add crossover point indicator plane
            if i >= 2 and layer_idx == crossover_point:  # Only for children at crossover point
                # Add a more visible plane at crossover point
                z_level = layer["z_level"]
                x = [0, pallet_dims.width, pallet_dims.width, 0]
                y = [0, 0, pallet_dims.length, pallet_dims.length]
                z = [z_level, z_level, z_level, z_level]
                verts = [list(zip(x, y, z))]
                rect = Poly3DCollection(verts, alpha=0.2, color="yellow")
                ax.add_collection3d(rect)

            for item_idx, item in enumerate(layer["items"]):
                coord = layer["coords"][item_idx]
                item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
                dims = utils.Dimension(item.width, item.length, item.height)

                # Determine color based on inheritance
                edgecolor = None
                alpha = 0.7
                linewidth = 1

                if i == 2:  # Child 1
                    item_ids = [item.id] if not isinstance(item.id, list) else item.id
                    if layer_idx < crossover_point:
                        # Before crossover point - from Parent 1
                        edgecolor = "#0000FF"  # Blue
                        linewidth = 1
                    else:
                        # After crossover point - from Parent 2
                        edgecolor = "#00AA00"  # Bright green
                        linewidth = 1
                elif i == 3:  # Child 2
                    item_ids = [item.id] if not isinstance(item.id, list) else item.id
                    if layer_idx < crossover_point:
                        # Before crossover point - from Parent 2
                        edgecolor = "#00AA00"  # Bright green
                        linewidth = 1
                    else:
                        # After crossover point - from Parent 1
                        edgecolor = "#0000FF"  # Blue
                        linewidth = 1

                # Add color hints to parent items too
                elif i == 0:  # Parent 1
                    item_ids = [item.id] if not isinstance(item.id, list) else item.id
                    edgecolor = "#0000FF"  # Blue
                    alpha = 0.5
                    linewidth = 1

                elif i == 1:  # Parent 2
                    item_ids = [item.id] if not isinstance(item.id, list) else item.id
                    edgecolor = "#00AA00"  # Bright green
                    alpha = 0.5
                    linewidth = 1

                # Plot with special highlighting if needed
                plot_product(
                    ax,
                    item_id,
                    coord,
                    dims,
                    pallet_dims,
                    alpha=0.9,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )

        # Add legend for children
        if i == 2 or i == 3:  # Child plots
            ax.text2D(
                0.05,
                0.95,
                "Blue border: From Parent 1",
                transform=ax.transAxes,
                fontsize=8,
                color="#0000FF",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7),
            )
            ax.text2D(
                0.05,
                0.90,
                "Green border: From Parent 2",
                transform=ax.transAxes,
                fontsize=8,
                color="#00AA00",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.7),
            )

    # After plotting, crop to max height used with small margin
    final_z_limit = all_max_z * 1.1 if all_max_z > 0 else pallet_dims.height
    for ax in axes:
        ax.set_zlim([0, final_z_limit])
        # Update box_aspect to match the cropped height
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, final_z_limit])

    plt.suptitle("Crossover Operation", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05)

    plt.show()
    return fig


def visualize_mutation(before_mutation, after_mutation, pallet_dims, mutation_type=None):
    """Visualize a chromosome before and after mutation with highlighted changes."""
    # Create figure with 1x2 grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

    chromosomes = [before_mutation, after_mutation]
    titles = ["Before Mutation", "After Mutation"]

    # Create item tracking dictionaries
    def create_item_map(chromosome):
        item_map = {}
        for layer_idx, layer in enumerate(chromosome):
            for item_idx, item in enumerate(layer["items"]):
                item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
                coord = layer["coords"][item_idx]
                item_map[item_id] = (layer_idx, item, coord)
        return item_map

    before_map = create_item_map(before_mutation)
    after_map = create_item_map(after_mutation)

    # Identify moved, added, and removed items
    moved_items = set()
    for item_id in before_map.keys() & after_map.keys():
        before_layer_idx, _, before_coord = before_map[item_id]
        after_layer_idx, _, after_coord = after_map[item_id]

        if (
            before_layer_idx != after_layer_idx
            or before_coord.x != after_coord.x
            or before_coord.y != after_coord.y
            or before_coord.z != after_coord.z
        ):
            moved_items.add(item_id)

    added_items = set(after_map.keys()) - set(before_map.keys())
    removed_items = set(before_map.keys()) - set(after_map.keys())

    # Detect layer changes
    before_layer_counts = len(before_mutation)
    after_layer_counts = len(after_mutation)
    layer_added = after_layer_counts > before_layer_counts
    layer_removed = after_layer_counts < before_layer_counts

    # Calculate max height used across both chromosomes for consistent display
    all_max_z = 0
    for chromosome in chromosomes:
        if chromosome:
            max_z = max(layer["z_level"] + layer["height"] for layer in chromosome)
            all_max_z = max(all_max_z, max_z)

    # Start with full pallet height for plotting
    initial_z_limit = pallet_dims.height

    # Plot each chromosome
    for i, (ax, chromosome, title) in enumerate(zip(axes, chromosomes, titles)):
        # Set axis limits and labels
        ax.set_xlim([0, pallet_dims.width])
        ax.set_ylim([0, pallet_dims.length])
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

        # Initially use full pallet height for plotting
        ax.set_zlim([0, initial_z_limit])

        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_zlabel("Height")

        # Disable gridlines and ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Add title with chromosome metrics
        num_layers = len(chromosome)
        total_items = sum(len(layer["items"]) for layer in chromosome)
        ax.set_title(f"{title}\nLayers: {num_layers}, Items: {total_items}")

        # Plot each layer with highlighted changes
        for layer_idx, layer in enumerate(chromosome):
            # Highlight changed layers
            if (layer_added and i == 1 and layer_idx == len(chromosome) - 1) or (
                layer_removed and i == 0 and layer_idx == len(chromosome) - 1
            ):
                # Highlight the affected layer
                z_level = layer["z_level"]
                x = [0, pallet_dims.width, pallet_dims.width, 0]
                y = [0, 0, pallet_dims.length, pallet_dims.length]
                z = [z_level, z_level, z_level, z_level]
                verts = [list(zip(x, y, z))]
                color = "green" if layer_added else "red"
                rect = Poly3DCollection(verts, alpha=0.15, color=color)
                ax.add_collection3d(rect)

            for item_idx, item in enumerate(layer["items"]):
                coord = layer["coords"][item_idx]
                item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
                dims = utils.Dimension(item.width, item.length, item.height)

                # Default appearance
                edgecolor = None
                alpha = 0.75
                linewidth = 1

                # Highlight based on change type
                if i == 0:  # Before mutation
                    if item_id in removed_items:
                        edgecolor = "red"  # Will be removed
                        linewidth = 1
                        alpha = 0.5
                    elif item_id in moved_items:
                        edgecolor = "orange"  # Will be moved
                        linewidth = 1
                        alpha = 0.5
                else:  # After mutation
                    if item_id in added_items:
                        edgecolor = "green"  # Newly added
                        linewidth = 1
                        alpha = 0.5
                    elif item_id in moved_items:
                        edgecolor = "orange"  # Was moved
                        linewidth = 1
                        alpha = 0.5

                # Plot with special highlighting
                plot_product(
                    ax,
                    item_id,
                    coord,
                    dims,
                    pallet_dims,
                    alpha=0.9,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )

        # Add legend
        if i == 0:  # Before mutation
            ax.text2D(
                0.05,
                0.95,
                "Red border: Will be removed",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
            ax.text2D(
                0.05,
                0.90,
                "Orange border: Will be moved",
                transform=ax.transAxes,
                fontsize=8,
                color="orange",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
        else:  # After mutation
            ax.text2D(
                0.05,
                0.95,
                "Green border: Newly added",
                transform=ax.transAxes,
                fontsize=8,
                color="green",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )
            ax.text2D(
                0.05,
                0.90,
                "Orange border: Was moved",
                transform=ax.transAxes,
                fontsize=8,
                color="orange",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.5),
            )

    # Add mutation type if provided
    mutation_desc = mutation_type if mutation_type else "Unknown"
    if layer_added:
        mutation_desc += " (Layer Added)"
    elif layer_removed:
        mutation_desc += " (Layer Removed)"

    plt.figtext(
        0.5,
        0.01,
        f"Mutation type: {mutation_desc}",
        ha="center",
        fontsize=12,
        bbox=dict(facecolor="yellow", alpha=0.2),
    )

    # Add change summary
    summary = (
        f"Changes: {len(moved_items)} moved, {len(added_items)} added, {len(removed_items)} removed"
    )
    plt.figtext(0.5, 0.05, summary, ha="center", fontsize=10)

    # After plotting, crop to max height used with small margin
    final_z_limit = all_max_z * 1.1 if all_max_z > 0 else pallet_dims.height
    for ax in axes:
        ax.set_zlim([0, final_z_limit])
        # Update box_aspect to match the cropped height
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, final_z_limit])

    plt.suptitle("Mutation Operation", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)

    return fig


def visualize_fitness_evolution(fitness_history, title="Fitness Evolution"):
    """Visualize the evolution of fitness over generations."""
    if not fitness_history:
        logger.warning("Empty fitness history, nothing to visualize")
        return None

    # Extract data from history
    generations = [entry["generation"] for entry in fitness_history]
    best_fitness = [entry["best"] for entry in fitness_history]
    avg_fitness = [entry.get("average", 0) for entry in fitness_history]
    worst_fitness = [entry.get("worst", 0) for entry in fitness_history]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot fitness values
    ax.plot(generations, best_fitness, "g-", linewidth=2, label="Best Fitness")
    ax.plot(generations, avg_fitness, "b-", linewidth=1.5, label="Average Fitness")
    ax.plot(generations, worst_fitness, "r-", linewidth=1, label="Worst Fitness")

    # Add horizontal line for current best fitness
    if best_fitness:
        ax.axhline(y=max(best_fitness), color="g", linestyle="--", alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Add annotations for best fitness
    if best_fitness:
        best_gen = generations[best_fitness.index(max(best_fitness))]
        ax.annotate(
            f"Best: {max(best_fitness):.4f}",
            xy=(best_gen, max(best_fitness)),
            xytext=(best_gen, max(best_fitness) * 0.9),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
            horizontalalignment="center",
        )

    plt.tight_layout()
    plt.show()

    return fig


def visualize_extreme_points(chromosome, layer_idx, pallet_dims):
    """Visualize the extreme points used in a layer for item placement."""
    if not chromosome or layer_idx >= len(chromosome):
        logger.warning(
            f"Invalid layer index {layer_idx} for chromosome of length {len(chromosome)}"
        )
        return None

    # Create pallet plot
    ax = get_pallet_plot(pallet_dims)

    # Get layer information
    layer = chromosome[layer_idx]
    z_level = layer["z_level"]
    max_z = z_level + layer["height"]

    # Update z-axis limits
    ax.set_zlim([z_level, max(max_z * 1.1, z_level + 100)])  # Add margin

    # Set title
    ax.set_title(f"Layer {layer_idx+1} with Extreme Points")

    # Plot items in the layer
    for item_idx, item in enumerate(layer["items"]):
        coord = layer["coords"][item_idx]
        item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
        dims = utils.Dimension(item.width, item.length, item.height)
        plot_product(ax, item_id, coord, dims, pallet_dims, alpha=0.75)

    # Generate extreme points
    used_positions = set()
    for item_idx, item in enumerate(layer["items"]):
        coord = layer["coords"][item_idx]
        used_positions.add((coord.x, coord.y, coord.x + item.width, coord.y + item.length))

    # Generate extreme points using the same logic as in the GA optimizer
    extreme_points = [(0, 0)]  # Start with origin

    # Add corner points from all used positions
    for x1, y1, x2, y2 in used_positions:
        extreme_points.extend(
            [
                (x1, y2),  # Bottom-right of current item becomes top-left of new
                (x2, y1),  # Top-right
                (x2, y2),  # Bottom-right
            ]
        )

    # Remove duplicates and filter valid points
    extreme_points = list(set(extreme_points))
    valid_points = []
    for x, y in extreme_points:
        if 0 <= x < pallet_dims.width and 0 <= y < pallet_dims.length:
            valid_points.append((x, y))

    # Plot extreme points
    for x, y in valid_points:
        ax.scatter(x, y, z_level, color="red", s=50, marker="x")
        ax.text(x, y, z_level, f"({x},{y})", fontsize=8, color="red")

    plt.tight_layout()
    plt.show()

    return plt.gcf()


def visualize_constraint_violations(chromosome, pallet_dims, title="Constraint Violations"):
    """Visualize constraint violations in a chromosome (overlaps, insufficient support)."""
    if not chromosome:
        logger.warning("Empty chromosome, nothing to visualize")
        return None

    # Create pallet plot
    ax = get_pallet_plot(pallet_dims)

    # Update z-axis limit based on actual solution height
    max_z = (
        max(layer["z_level"] + layer["height"] for layer in chromosome)
        if chromosome
        else pallet_dims.height
    )
    ax.set_zlim([0, max(max_z * 1.1, pallet_dims.height)])  # Add margin

    # Set title
    ax.set_title(title)

    # Track violations
    overlaps = []
    insufficient_support = []

    # First pass: collect all items and identify violations
    all_items = []
    for layer_idx, layer in enumerate(chromosome):
        z_level = layer["z_level"]

        for item_idx, item in enumerate(layer["items"]):
            coord = layer["coords"][item_idx]
            all_items.append((layer_idx, item_idx, item, coord))

    # Check for overlaps
    for i, (layer1_idx, item1_idx, item1, coord1) in enumerate(all_items):
        for j, (layer2_idx, item2_idx, item2, coord2) in enumerate(all_items[i + 1 :], i + 1):
            # Check if bounding boxes overlap
            x_overlap = max(
                0, min(coord1.x + item1.width, coord2.x + item2.width) - max(coord1.x, coord2.x)
            )
            y_overlap = max(
                0, min(coord1.y + item1.length, coord2.y + item2.length) - max(coord1.y, coord2.y)
            )
            z_overlap = max(
                0, min(coord1.z + item1.height, coord2.z + item2.height) - max(coord1.z, coord2.z)
            )

            if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
                overlaps.append((i, j))

    # Check for insufficient support
    for layer_idx, layer in enumerate(chromosome):
        z_level = layer["z_level"]

        # Skip items on ground level (z=0)
        if z_level == 0:
            continue

        for item_idx, item in enumerate(layer["items"]):
            coord = layer["coords"][item_idx]

            # Get items in the layer below for support check
            supported_area = 0
            item_base_area = item.width * item.length

            # Find supporting items
            for other_layer_idx, other_layer in enumerate(chromosome):
                if other_layer["z_level"] + other_layer["height"] == z_level:  # Direct support
                    for other_item_idx, other_item in enumerate(other_layer["items"]):
                        other_coord = other_layer["coords"][other_item_idx]

                        # Calculate overlap area
                        x_overlap = max(
                            0,
                            min(coord.x + item.width, other_coord.x + other_item.width)
                            - max(coord.x, other_coord.x),
                        )
                        y_overlap = max(
                            0,
                            min(coord.y + item.length, other_coord.y + other_item.length)
                            - max(coord.y, other_coord.y),
                        )

                        if x_overlap > 0 and y_overlap > 0:
                            supported_area += x_overlap * y_overlap

            # Calculate support percentage
            support_percentage = supported_area / item_base_area if item_base_area > 0 else 0

            # Check if support is insufficient (less than 70%)
            if support_percentage < 0.7:
                insufficient_support.append((layer_idx, item_idx))

    # Second pass: plot items with violations highlighted
    for layer_idx, layer in enumerate(chromosome):
        for item_idx, item in enumerate(layer["items"]):
            coord = layer["coords"][item_idx]
            item_id = item.id if not isinstance(item.id, list) else tuple(item.id)
            dims = utils.Dimension(item.width, item.length, item.height)

            # Check if this item has violations
            has_overlap = any(
                i == all_items.index((layer_idx, item_idx, item, coord))
                or j == all_items.index((layer_idx, item_idx, item, coord))
                for i, j in overlaps
            )

            has_support_issue = (layer_idx, item_idx) in insufficient_support

            if has_overlap:
                # Highlight items with overlap violations
                plot_product(
                    ax, item_id, coord, dims, pallet_dims, alpha=0.5, edgecolor="red", linewidth=2
                )

                # Add text label with violation type
                center = utils.Vertices(coord, dims).get_center()
                ax.text(
                    center.x,
                    center.y,
                    center.z,
                    "Overlap",
                    color="red",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

            elif has_support_issue:
                # Highlight items with support violations
                plot_product(
                    ax,
                    item_id,
                    coord,
                    dims,
                    pallet_dims,
                    alpha=0.5,
                    edgecolor="orange",
                    linewidth=2,
                )

                # Add text label with violation type
                center = utils.Vertices(coord, dims).get_center()
                ax.text(
                    center.x,
                    center.y,
                    center.z,
                    "Support",
                    color="orange",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

            else:
                # Normal item
                plot_product(ax, item_id, coord, dims, pallet_dims, alpha=0.75)

    # Add summary text
    violation_text = (
        f"Violations - Overlaps: {len(overlaps)}, Support Issues: {len(insufficient_support)}"
    )
    ax.text2D(
        0.5,
        0.95,
        violation_text,
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    return plt.gcf()
