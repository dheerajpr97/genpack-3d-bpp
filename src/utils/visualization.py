import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Global dictionary to store item_id to color mapping
item_colors = {}

# Predefined color palette (e.g., tab20) for distinct colors
color_palette = plt.cm.tab20(np.linspace(0, 1, 20))  # Use 20 predefined colors


# Function to assign or retrieve color for an item_id
def get_item_color(item_id):
    """Assign or retrieve color for an item_id."""
    if item_id not in item_colors:
        # Assign a new color if not already assigned
        new_color = color_palette[len(item_colors) % len(color_palette)]
        item_colors[item_id] = new_color
    return item_colors[item_id]


# Function to create a pallet plot
def get_pallet_plot(pallet_dims, figsize=(10, 8)):
    """Create a 3D plot for the pallet."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

    # Set axis limits
    ax.set_xlim([0, pallet_dims.width])
    ax.set_ylim([0, pallet_dims.length])
    ax.set_zlim([0, pallet_dims.height])

    # Label axes
    ax.set_xlabel("Width")
    ax.set_ylabel("Length")
    ax.set_zlabel("Height")

    return ax


# Function to plot products
def plot_product(
    ax, item_id, coords, dims, pallet_dims, alpha=0.75, edgecolor="red", linewidth=0.25
):
    """Add product to given axis."""

    from src.utils.utils import Vertices

    vertices = Vertices(coords, dims)

    poly3d = Poly3DCollection(
        vertices.to_faces(),
        facecolors=get_item_color(item_id),
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

    ax.add_collection3d(poly3d)

    center = vertices.get_center()
    ax.text(
        center.x,
        center.y,
        center.z,
        str(item_id),
        size=10,
        zorder=1,
        color="k",
        ha="center",
        va="center",
    )
    return ax


# Function to plot superitems
def plot_superitems(superitems, pallet_dims, cols=3):
    """Plot superitems in a grid layout."""
    num_superitems = len(superitems)
    rows = (num_superitems + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6, rows * 6), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten()

    for idx, superitem in enumerate(superitems):
        ax = axes[idx]
        ax.set_title(f"Superitem {idx + 1}", fontsize=8, loc="center", y=0.9)
        ax.set_xlim([0, pallet_dims.width])
        ax.set_ylim([0, pallet_dims.length])
        ax.set_zlim([0, pallet_dims.height])
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

        item_coords = superitem.get_items_coords()
        item_dims = superitem.get_items_dims()

        for item_id, coord in item_coords.items():
            dims = item_dims[item_id]
            plot_product(ax, item_id, coord, dims, pallet_dims, alpha=0.3)

    for ax in axes[num_superitems:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
    return fig


def plot_height_groups(height_groups, pallet_dims, cols=3):
    """Visualize all height groups as a grid of 3D plots using stacking logic."""
    from src.utils.utils import Coordinate

    num_groups = len(height_groups)
    if num_groups == 0:
        logger.info("No height groups to visualize.")
        return None

    rows = (num_groups + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6, rows * 6), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten()

    for idx, group in enumerate(height_groups):
        ax = axes[idx]
        ax.set_title(f"Height Group {idx + 1}", fontsize=8, loc="center", y=0.9)
        ax.set_xlim([0, pallet_dims.width])
        ax.set_ylim([0, pallet_dims.length])
        ax.set_zlim([0, pallet_dims.height])
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])

        # Track stacking height for each XY location
        stacking_heights = {}

        for superitem in group:
            item_coords = superitem.get_items_coords()
            item_dims = superitem.get_items_dims()

            for item_id, coord in item_coords.items():
                dims = item_dims[item_id]

                # Get the stacking height for this (x, y) position
                xy_key = (coord.x, coord.y)
                base_z = stacking_heights.get(xy_key, 0)  # Default to 0 if no prior stacking
                stacking_heights[xy_key] = base_z + dims.height  # Update stacking height

                # Assign the correct z-position
                adjusted_coord = Coordinate(coord.x, coord.y, base_z)

                # Plot the product
                plot_product(ax, item_id, adjusted_coord, dims, pallet_dims, alpha=0.3)

    for ax in axes[num_groups:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
    return fig


# Function to plot layers
def plot_layers(layer_pool, pallet_dims, cols=6, title=None):
    """Plot layers in a grid layout."""
    num_layers = len(layer_pool)
    if num_layers == 0:
        logger.info("No layers to visualize.")
        return None

    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 6, rows * 6), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten()

    for i, layer in enumerate(layer_pool):
        ax = axes[i]

        if title is not None:
            ax.set_title(title, fontsize=8, y=1.05)
        else:
            ax.set_title(
                f"Layer {i + 1}\nHeight: {layer.height}\nDensity: {layer.get_density():.2f}",
                fontsize=7,
                y=0.95,
            )

        ax.set_xlim(0, pallet_dims.width)
        ax.set_ylim(0, pallet_dims.length)
        ax.set_zlim(0, pallet_dims.height)
        ax.set_box_aspect([pallet_dims.width, pallet_dims.length, pallet_dims.height])
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=6)

        items_coords = layer.get_items_coords()
        items_dims = layer.get_items_dims()

        for item_id, coords in items_coords.items():
            dims = items_dims[item_id]
            plot_product(ax, item_id, coords, dims, pallet_dims, alpha=0.25)

    for j in range(num_layers, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return fig


# Function to visualize pre and post filter pools
def visualize_pre_post_filter(pre_filter_pool, post_filter_pool, pallet_dims, cols=2):
    """Visualize pre and post filter pools."""
    logger.info("Visualizing layers before filtering...")
    if len(pre_filter_pool) > 0:
        logger.info(f"Pre-filter pool contains {len(pre_filter_pool)} layers.")
        pre_fig = plot_layers(
            pre_filter_pool, pallet_dims, cols, title="Before Filtering\n(on pooled layers)"
        )
    else:
        logger.info("No pre-filter layers to visualize.")
        pre_fig = None

    logger.info("\nVisualizing layers after filtering...")
    if len(post_filter_pool) > 0:
        logger.info(f"Post-filter pool contains {len(post_filter_pool)} layers.")
        post_fig = plot_layers(post_filter_pool, pallet_dims, cols, title="After Filtering")
    else:
        logger.info("No post-filter layers to visualize.")
        post_fig = None

    return pre_fig, post_fig


def export_bin_packing_to_csv(bins, output_file="bin_packing_results.csv"):
    """Export bin packing results to CSV format compatible with the web visualizer."""
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["container_id", "item_id", "x", "y", "z", "width", "length", "height"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through your bin structure
        for bin_idx, bin_items in enumerate(bins):
            container_id = f"bin_{bin_idx}"

            # Iterate through items in this bin
            for item_id, data in bin_items.items():
                coords = data["coords"]  # Assuming this is your coordinate object
                dims = data["dims"]  # Assuming this is your dimension object

                writer.writerow(
                    {
                        "container_id": container_id,
                        "item_id": item_id,
                        "x": coords.x,
                        "y": coords.y,
                        "z": coords.z,
                        "width": dims.width,
                        "length": dims.length,
                        "height": dims.height,
                    }
                )

    logger.info(f"Exported bin packing results to {output_file}")
    return output_file


def set_common_labels_and_colors(axes_list, pallet_dims):
    """Set common labels, colors, and viewing angles for multiple 3D axes."""
    for ax in axes_list:
        # Set axis labels
        ax.set_xlabel("Width", fontsize=10)
        ax.set_ylabel("Length", fontsize=10)
        ax.set_zlabel("Height", fontsize=10)

        # Draw pallet outline
        ax.plot(
            [0, pallet_dims.width, pallet_dims.width, 0, 0],
            [0, 0, pallet_dims.length, pallet_dims.length, 0],
            [0, 0, 0, 0, 0],
            "k-",
            linewidth=1,
        )

        # Draw vertical lines at corners
        for x, y in [
            (0, 0),
            (pallet_dims.width, 0),
            (pallet_dims.width, pallet_dims.length),
            (0, pallet_dims.length),
        ]:
            ax.plot([x, x], [y, y], [0, pallet_dims.height], "k-", linewidth=1, alpha=0.3)

        # Set a common viewing angle for better comparison
        ax.view_init(elev=30, azim=45)

        # Add grid for better depth perception
        ax.grid(True, alpha=0.3)

        # Make axes ticks more readable
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontsize(8)
