import numpy as np
import pandas as pd
from loguru import logger

from src import config
from src.models import layers, maxrects, superitems
from src.utils import utils, visualization


class Bin:
    """Represent a single pallet bin composed of stacked layers."""

    def __init__(self, layer_pool, pallet_dims):
        """Initialize a bin."""
        self.layer_pool = layer_pool
        self.pallet_dims = pallet_dims

    @property
    def height(self):
        """Return the total stacked height of the bin."""
        return sum(l.height for l in self.layer_pool)

    @property
    def volume(self):
        """Return the packed item volume in the bin."""
        return sum(l.volume for l in self.layer_pool)

    @property
    def remaining_height(self):
        """Return the remaining vertical space in the bin."""
        return self.pallet_dims.height - self.height

    def add(self, layer):
        """Add a layer to the bin and assign its vertical position."""
        assert isinstance(
            layer, layers.Layer
        ), "The given layer should be an instance of the Layer class"
        # Calculate base_z as the cumulative height of layers in this bin, capped at pallet height
        base_z = min(sum(l.height for l in self.layer_pool), self.pallet_dims.height)
        # Reset base_z to 0 if it exceeds expected stacking (debug to identify inflation)
        expected_base_z = sum(
            l.height for l in self.layer_pool if l.height <= self.pallet_dims.height
        )
        base_z = min(base_z, expected_base_z, self.pallet_dims.height)
        logger.debug(f"Adding layer to bin, base_z={base_z}, expected_base_z={expected_base_z}")
        layer.assign_coordinates(base_z=base_z)  # Assign z based on stacking
        self.layer_pool.add(layer)

    def get_layer_zs(self):
        """Return the starting z-coordinate for each layer."""
        heights = [0]
        for layer in self.layer_pool[:-1]:
            heights += [heights[-1] + layer.height]
        return heights

    def get_layer_densities(self, two_dims=False):
        """Return density values for each layer in the bin."""
        return self.layer_pool.get_densities(two_dims=two_dims)

    def get_density(self):
        """Return the overall volume utilization of the bin."""
        return self.volume / self.pallet_dims.volume

    def sort_by_densities(self, two_dims=False):
        """Sort layers in descending density order."""
        self.layer_pool.sort_by_densities(two_dims=two_dims)

    def plot(self):
        """Plot the bin as a 3D stacked layout."""
        height = 0
        ax = visualization.get_pallet_plot(self.pallet_dims)
        for layer in self.layer_pool:
            ax = layer.plot(ax=ax, height=height)
            height += layer.height
        return ax

    def to_dataframe(self):
        """Convert the bin contents to a pandas DataFrame."""
        dfs = []
        for layer_idx, layer in enumerate(self.layer_pool):
            df = layer.to_dataframe()
            df["layer"] = layer_idx  # Add layer index
            df["bin"] = 0
            dfs.append(df)
        if not dfs:
            return pd.DataFrame(
                columns=[
                    "item",
                    "x",
                    "y",
                    "z",
                    "width",
                    "length",
                    "height",
                    "weight",
                    "layer",
                    "bin",
                ]
            )
        return pd.concat(dfs, axis=0)

    def __str__(self):
        """Return a compact string representation of the bin."""
        return f"Bin({self.layer_pool})"

    def __repr__(self):
        """Return the developer-facing string representation of the bin."""
        return self.__str__()


class BinPool:
    """Manage a collection of pallet bins for a packing solution."""

    def __init__(self, layer_pool, pallet_dims, singles_removed=None, two_dims=False, area_tol=1.0):
        """Initialize a pool of bins and distribute layers across them."""
        self.layer_pool = layer_pool
        self.pallet_dims = pallet_dims

        # Build the bin pool and place uncovered items on top
        # or in a new bin
        self.bins = self._build(self.layer_pool)

        # Sort layers in each bin by density
        for bin in self.bins:
            bin.sort_by_densities(two_dims=two_dims)

    def _build(self, layer_pool):
        """Build bins by distributing layers based on height constraints."""
        bins = []
        for i, layer in enumerate(layer_pool):
            placed = False
            for bin in bins:
                if bin.height + layer.height <= self.pallet_dims.height:
                    bin.add(layer)
                    placed = True
            if not placed:
                bins += [Bin(layer_pool.subset([i]), self.pallet_dims)]
        return bins

    def get_heights(self):
        """Get the current height of each bin in the pool."""
        return [b.height for b in self.bins]

    def get_remaining_heights(self):
        """Get the remaining vertical space in each bin."""
        return [b.remaining_height for b in self.bins]

    def get_layer_densities(self, two_dims=False):
        """Get density metrics for all layers across all bins."""
        return [b.get_layer_densities(two_dims) for b in self.bins]

    def get_bin_densities(self):
        """Get the overall density for each bin in the pool."""
        return [b.get_density() for b in self.bins]

    def plot(self):
        """Generate 3D visualizations for all bins in the pool."""
        axs = []
        for bin in self.bins:
            ax = bin.plot()
            ax.set_facecolor("xkcd:white")
            axs.append(ax)
        return axs

    def to_dataframe(self):
        """Convert all bins in the pool to a single pandas DataFrame."""
        dfs = []
        for bin_idx, bin in enumerate(self.bins):
            df = bin.to_dataframe()
            df["bin"] = bin_idx  # Already correct
            dfs.append(df)
        if not dfs:
            return pd.DataFrame(
                columns=[
                    "item",
                    "x",
                    "y",
                    "z",
                    "width",
                    "length",
                    "height",
                    "weight",
                    "layer",
                    "bin",
                ]
            )
        return pd.concat(dfs, axis=0)

    def __str__(self):
        """Return string representation of the bin pool."""
        return f"BinPool(bins={self.bins})"

    def __repr__(self):
        """Return detailed string representation of the bin pool."""
        return self.__str__()

    def __len__(self):
        """Return the number of bins in the pool."""
        return len(self.bins)

    def __contains__(self, bin):
        """Check if a bin is contained in the pool."""
        return bin in self.bins

    def __getitem__(self, i):
        """Get a bin by index."""
        return self.bins[i]

    def __setitem__(self, i, e):
        """Set a bin at the specified index."""
        assert isinstance(e, Bin), "The given bin should be an instance of the Bin class"
        self.bins[i] = e


class CompactBin:
    """Advanced bin implementation with optimization and validation capabilities."""

    def __init__(
        self,
        bin_df,
        pallet_dims,
        use_sequential=True,
        validate_final=True,
        optimize_height=True,
        enable_void_creation=True,
        compaction_strength="normal",
    ):
        """Initialize the compact bin with advanced optimization capabilities."""
        self.pallet_dims = pallet_dims
        self.bin_df = bin_df.copy()
        self.use_sequential = use_sequential
        self.optimize_height = optimize_height
        self.enable_void_creation = enable_void_creation
        self.compaction_strength = compaction_strength

        logger.info(f"Initializing CompactBin with {len(self.bin_df)} items")
        logger.info(
            f"Pallet dimensions: W={pallet_dims.width}, L={pallet_dims.length}, H={pallet_dims.height}"
        )
        logger.info(
            f"Configuration: Sequential={use_sequential}, Optimize Height={optimize_height}, "
            + f"Enable Void Creation={enable_void_creation}, Compaction={compaction_strength}"
        )

        # Process and optimize the bin layer by layer if using sequential approach
        if self.use_sequential:
            self.df = self._optimize_layer_placement(self.bin_df.reset_index())
        else:
            # If not using sequential approach, just use the original dataframe
            self.df = self.bin_df.copy()

        # Perform final validation if requested
        self.validated_df = self.df  # Default to unvalidated solution
        if validate_final:
            self.validated_df, self.validation_report = self._validate_final_packing()

    def _optimize_high_items(
        self, validated_df=None, z_threshold_ratio=0.6, max_items_to_check=10, max_iterations=3
    ):
        """Attempt to relocate items from the highest z-levels to lower positions in the bin."""
        # Use validated dataframe if available, otherwise use the original dataframe
        df = validated_df if validated_df is not None else self.df.copy()
        if df.empty:
            return df

        # Track relocated items across all iterations
        all_relocated_items = set()

        # Keep track of iterations to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get maximum z height in the bin
            max_z_height = max(row.z + row.height for _, row in df.iterrows())

            # Determine the z threshold above which items are considered "high"
            z_threshold = max_z_height * z_threshold_ratio

            # Find items above the threshold
            high_items = df[df.z + df.height > z_threshold].copy()

            # Filter out items that have already been relocated in previous iterations
            high_items = high_items[~high_items["item"].isin(all_relocated_items)]

            # If no high items remain to be relocated, we're done
            if high_items.empty:
                logger.info(f"No more high items to relocate after {iteration} iterations")
                break

            # Sort high items by z-coordinate (highest first)
            high_items = high_items.sort_values(by=["z"], ascending=False)

            # Limit the number of items to check to reduce computational overhead
            high_items = high_items.iloc[:max_items_to_check]

            logger.info(
                f"Iteration {iteration}: Attempting to relocate {len(high_items)} items above z={z_threshold:.2f}"
            )

            # Track items relocated in this iteration
            items_relocated_this_iteration = set()

            # Process each high item
            for idx, item in high_items.iterrows():
                # Skip if this item has already been relocated in this iteration
                if item["item"] in items_relocated_this_iteration:
                    continue

                # Get all other items (excluding the current high item)
                other_items = df[df.item != item["item"]].copy()
                other_items_list = other_items.to_dict("records")

                # Store original position to check if something actually changed
                original_position = (item["x"], item["y"], item["z"])

                # Try to find a better position for this item
                better_position = self._find_lower_position(item, other_items_list, max_z_height)

                # If a better position is found, update the item's position
                if better_position:
                    # Check if the position actually changed
                    new_position = (
                        better_position["x"],
                        better_position["y"],
                        better_position["z"],
                    )
                    if new_position != original_position:
                        # Update the dataframe
                        df.loc[idx, "x"] = better_position["x"]
                        df.loc[idx, "y"] = better_position["y"]
                        df.loc[idx, "z"] = better_position["z"]

                        logger.info(
                            f"Relocated item {item['item']} from {original_position} to "
                            + f"({better_position['x']}, {better_position['y']}, {better_position['z']})"
                        )

                        # Mark this item as relocated
                        items_relocated_this_iteration.add(item["item"])
                        all_relocated_items.add(item["item"])
                    else:
                        logger.debug(
                            f"Found position for item {item['item']} but it's the same as current position"
                        )

            # If no items were relocated in this iteration, we're done
            if not items_relocated_this_iteration:
                logger.info(f"No items relocated in iteration {iteration} - stopping optimization")
                break

            # Validate the resulting solution to ensure no constraint violations
            if items_relocated_this_iteration:
                logger.info(
                    f"Successfully relocated {len(items_relocated_this_iteration)} high items in iteration {iteration}"
                )

                # Perform validation to ensure the new solution is valid
                validated_df, validation_report = self._validate_final_packing(df)

                # Check if validation removed items
                if len(validation_report["removed_items"]) > 0:
                    logger.warning(
                        f"Validation removed {len(validation_report['removed_items'])} items "
                        + f"after high item optimization. Reverting to original solution."
                    )
                    return self.df.copy()

                # Update our working df with the validated one
                df = validated_df

        # Log if we reached maximum iterations
        if iteration == max_iterations:
            logger.warning(
                f"Reached maximum {max_iterations} iterations for high item optimization"
            )

        return df

    def _find_lower_position(self, item, other_items, current_max_height):
        """Try to find a lower position for a high item."""
        # Extract item dimensions
        width, length, height = item["width"], item["length"], item["height"]
        current_z = item["z"]

        # Define search parameters for computational efficiency
        step_size = 50
        min_support_threshold = 0.75  # Require good support for relocated items

        # Initialize best position variables
        best_position = None
        best_z = current_z  # Start with current z as baseline

        # Find all surface levels where this item could potentially be placed
        surface_levels = set([0])  # Always include ground level
        for other_item in other_items:
            # Add the top surface of each item as a potential placement level
            surface_levels.add(other_item["z"] + other_item["height"])

        # Sort surface levels for efficient search (lowest first)
        surface_levels = sorted(surface_levels)

        # Only consider surface levels below the current item's position
        surface_levels = [z for z in surface_levels if z < current_z]

        # For each potential surface level, try to place the item
        for z_level in surface_levels:
            # Skip if this would make the item exceed the pallet height
            if z_level + height > self.pallet_dims.height:
                continue

            # Try positions at this z-level with grid search
            for x in range(0, int(self.pallet_dims.width - width + 1), step_size):
                for y in range(0, int(self.pallet_dims.length - length + 1), step_size):
                    # Skip if position isn't valid
                    if not self._is_within_pallet_bounds(x, y, z_level, width, length, height):
                        continue

                    # Skip if overlaps with other items
                    if self._check_overlap(x, y, z_level, width, length, height, other_items):
                        continue

                    # Calculate support percentage
                    support_percentage, is_supported = self._calculate_support(
                        x, y, z_level, width, length, other_items, min_support_threshold
                    )

                    if is_supported:
                        # Calculate new max height if this item is moved
                        new_max_height = max(
                            z_level + height,
                            max(
                                other_item["z"] + other_item["height"] for other_item in other_items
                            ),
                        )

                        # Only accept if this improves overall height or keeps same height with better support
                        if new_max_height < current_max_height or (
                            new_max_height == current_max_height and z_level < best_z
                        ):
                            best_z = z_level
                            best_position = {
                                "x": x,
                                "y": y,
                                "z": z_level,
                                "support": support_percentage,
                            }

                            # If the new position is at ground level with full support,
                            # we can immediately return as this is optimal
                            if z_level == 0:
                                return best_position

        return best_position

    # Modified _optimize_layer_placement method to prioritize existing z-levels
    def _optimize_layer_placement(self, bin_df):
        """Optimize item placement layer by layer to create a stable,."""
        # Create a working copy of the dataframe
        df = bin_df.copy()

        # Add a processing status column
        df["processed"] = False

        # Get all unique layers in ascending order
        if "layer" in df.columns:
            layers = sorted(df["layer"].unique())
        else:
            # If no layer information, create synthetic layers based on z-coordinate
            df["layer"] = df["z"].apply(lambda z: int(z / 200))  # Arbitrary height of 200 per layer
            layers = sorted(df["layer"].unique())

        logger.info(f"Processing {len(layers)} layers: {layers}")

        # Process layer 0 items
        layer0_items = df[df["layer"] == layers[0]].copy()
        if not layer0_items.empty:
            layer0_items = self._optimize_layer0_items(layer0_items)
            # Update the processed items in the main dataframe
            for idx in layer0_items.index:
                df.loc[idx, "x"] = layer0_items.loc[idx, "x"]
                df.loc[idx, "y"] = layer0_items.loc[idx, "y"]
                df.loc[idx, "z"] = layer0_items.loc[idx, "z"]  # Preserve original z-coordinate
                df.loc[idx, "processed"] = True

            # Apply compaction to layer 0 if enabled
            if hasattr(self, "enable_void_creation") and self.enable_void_creation:
                force_clustering = False  # Normal compaction doesn't use aggressive clustering
                df = self._compact_layer_items(df, layers[0], force_clustering)

        # Collect and maintain a sorted list of all z-levels in the bin as items are placed
        existing_z_levels = sorted(df[df["processed"]]["z"].unique().tolist())
        if not existing_z_levels and not layer0_items.empty:
            # If no z-levels yet, initialize with layer0
            existing_z_levels = [0]

        # Process subsequent layers
        for layer_idx in range(1, len(layers)):
            current_layer = layers[layer_idx]
            layer_items = df[df["layer"] == current_layer].copy()

            logger.info(
                f"Processing layer {current_layer}/{layers[-1]} with {len(layer_items)} items"
            )

            # Sort layer items by base area (larger footprint first)
            layer_items["base_area"] = layer_items["width"] * layer_items["length"]
            layer_items = layer_items.sort_values("base_area", ascending=False)

            # Keep track of unplaced items
            unplaced_items = []

            # Process each item in the current layer
            for idx in layer_items.index:
                item = layer_items.loc[idx]
                processed_items = df[df["processed"]].copy()

                # Flag to track if the item has been placed successfully
                item_placed = False

                # 1. First try to place the item at existing z-levels before creating a new one
                for z_level in existing_z_levels:
                    position = self._find_position_at_existing_z(item, processed_items, z_level)

                    if position:
                        # Update item position and mark as processed
                        df.loc[idx, "x"] = position["x"]
                        df.loc[idx, "y"] = position["y"]
                        df.loc[idx, "z"] = position["z"]
                        df.loc[idx, "processed"] = True
                        item_placed = True

                        logger.info(
                            f"Placed item {item['item']} at existing z-level {z_level} at coordinates ({position['x']}, {position['y']}, {position['z']})"
                        )
                        break

                # If the item couldn't be placed at existing z-levels, proceed with normal placement strategies
                if not item_placed:
                    # Normal interlocking approach
                    position = self._find_interlocking_position(item, processed_items)
                    if position:
                        placement_type = "interlocking"
                    else:
                        position = None

                    # If no position found yet, try other strategies
                    if not position:
                        # Fall back to existing strategies
                        position = self._find_optimal_position(
                            item,
                            processed_items,
                            current_layer,
                            max(existing_z_levels) if existing_z_levels else 0,
                        )

                        if position:
                            placement_type = "optimal"
                        else:
                            position = self._find_fallback_position(item, processed_items)
                            if position:
                                placement_type = "fallback"
                            else:
                                position = self._find_alternative_position(item, processed_items)
                                if position:
                                    placement_type = "alternative"

                                    # Handle rotation if item was rotated
                                    if position.get("rotated", False):
                                        placement_type += " (rotated)"
                                        # Update the item dimensions in the layer_items dataframe
                                        if "width" in position:
                                            layer_items.loc[idx, "width"] = position["width"]
                                        if "length" in position:
                                            layer_items.loc[idx, "length"] = position["length"]

                                        logger.info(
                                            f"Item {item['item']} rotated: new dimensions {position.get('width', item['width'])}x{position.get('length', item['length'])}x{item['height']}"
                                        )

                                    else:
                                        unplaced_items.append(idx)
                                        logger.info(
                                            f"Could not find position for item {item['item']} - will try fallback"
                                        )
                                        continue

                    # Update item position and mark as processed
                    if position:
                        # CRITICAL: Validate the position before accepting it
                        test_x, test_y, test_z = position["x"], position["y"], position["z"]
                        test_width = position.get("width", item["width"])
                        test_length = position.get("length", item["length"])
                        test_height = item["height"]

                        # Get all other processed items for validation
                        other_items = df[df["processed"] & (df["item"] != item["item"])].copy()
                        other_items_list = other_items.to_dict("records")

                        # Validate the position
                        is_valid_position, validation_errors = (
                            self._validate_position_during_placement(
                                test_x,
                                test_y,
                                test_z,
                                test_width,
                                test_length,
                                test_height,
                                other_items_list,
                                0.3,
                            )
                        )

                        # If position is invalid, reject it and continue to next strategy
                        if not is_valid_position:
                            logger.warning(
                                f"Rejected invalid position for item {item['item']} at ({test_x}, {test_y}, {test_z}): {', '.join(validation_errors)}"
                            )
                            position = None
                            continue

                        # Position is valid, proceed with placement
                        df.loc[idx, "x"] = position["x"]
                        df.loc[idx, "y"] = position["y"]
                        df.loc[idx, "z"] = position["z"]

                        # Update dimensions if item was rotated
                        if position.get("rotated", False):
                            if "width" in position:
                                df.loc[idx, "width"] = position["width"]
                            if "length" in position:
                                df.loc[idx, "length"] = position["length"]

                        df.loc[idx, "processed"] = True

                        logger.info(
                            f"{placement_type.capitalize()}: Placed item {item['item']} at ({position['x']}, {position['y']}, {position['z']})"
                        )

                        # Update existing z-levels if a new one was created
                        if position["z"] not in existing_z_levels:
                            existing_z_levels.append(position["z"])
                            existing_z_levels.sort()  # Keep the list sorted
                    else:
                        logger.warning(
                            f"Could not find any valid position for item {item['item']} in layer {current_layer} - item will remain unplaced."
                        )
                        # Optionally, mark as unplaced or handle as needed

            # After placing all items in this layer, compact them to create denser packing
            if (
                not layer_items.empty
                and hasattr(self, "enable_void_creation")
                and self.enable_void_creation
            ):
                # Normal compaction - no clustering
                use_clustering = False
                df = self._compact_layer_items(df, current_layer, use_clustering)

            # Handle any remaining unplaced items
            if unplaced_items:
                logger.warning(
                    f"Could not place {len(unplaced_items)} items in layer {current_layer} - attempting alternative placement"
                )

                # For remaining unplaced items, try to find any valid position at any z-level
                logger.info(f"Attempting final placement for {len(unplaced_items)} unplaced items")

                for idx in unplaced_items[:]:
                    item = layer_items.loc[idx]
                    processed_items = df[df["processed"]].copy()

                    # Try alternative position with very relaxed constraints
                    position = self._find_alternative_position(item, processed_items)

                    if position:
                        # Validate position before accepting
                        test_x, test_y, test_z = position["x"], position["y"], position["z"]
                        test_width = position.get("width", item["width"])
                        test_length = position.get("length", item["length"])
                        test_height = item["height"]

                        other_items = df[df["processed"] & (df["item"] != item["item"])].copy()
                        other_items_list = other_items.to_dict("records")

                        # Validate the position
                        is_valid_position, validation_errors = (
                            self._validate_position_during_placement(
                                test_x,
                                test_y,
                                test_z,
                                test_width,
                                test_length,
                                test_height,
                                other_items_list,
                                0.1,  # Very relaxed
                            )
                        )

                        if is_valid_position:
                            # Update item position and mark as processed
                            df.loc[idx, "x"] = position["x"]
                            df.loc[idx, "y"] = position["y"]
                            df.loc[idx, "z"] = position["z"]

                            if position.get("rotated", False):
                                if "width" in position:
                                    df.loc[idx, "width"] = position["width"]
                                if "length" in position:
                                    df.loc[idx, "length"] = position["length"]

                            df.loc[idx, "processed"] = True

                            unplaced_items.remove(idx)
                            logger.info(
                                f"Final placement: Item {item['item']} at ({position['x']}, {position['y']}, {position['z']})"
                            )
                        else:
                            logger.warning(
                                f"Final position for item {item['item']} failed validation: {', '.join(validation_errors)}"
                            )

                if unplaced_items:
                    logger.error(
                        f"Could not place {len(unplaced_items)} items even with final attempts - these items will be excluded"
                    )
                    # Mark unplaced items as not processed so they won't be included in final result
                    for idx in unplaced_items:
                        df.loc[idx, "processed"] = False

        # Final post-processing pass for all layers
        # Normal compaction - no additional analysis needed for void creation

        # Remove temporary columns before returning
        columns_to_drop = ["processed"]
        if "base_area" in df.columns:
            columns_to_drop.append("base_area")

        # CRITICAL: Only return items that were successfully processed
        final_df = df[df["processed"] == True].copy()

        if len(final_df) < len(df):
            logger.warning(
                f"Excluded {len(df) - len(final_df)} unplaced/invalid items from final result"
            )
            logger.info(
                f"Final result contains {len(final_df)} valid items out of {len(df)} total items"
            )

        return final_df.drop(columns=columns_to_drop)

    def _fill_layer_voids(self, df, current_layer):
        """Identify and fill voids in the current layer to maximize density."""
        # Create a working copy
        filled_df = df.copy()

        # Get all items in the current layer
        layer_mask = (filled_df["layer"] == current_layer) & (filled_df["processed"] == True)
        layer_items = filled_df[layer_mask].copy()

        if layer_items.empty:
            return filled_df

        logger.info(f"Analyzing voids in layer {current_layer} for potential filling")

        # Get layer bounds
        min_x = layer_items["x"].min()
        min_y = layer_items["y"].min()
        max_x = layer_items["x"].max() + layer_items["width"].max()
        max_y = layer_items["y"].max() + layer_items["length"].max()
        z_level = layer_items["z"].mean()  # Approximate z-level

        # Create a grid for void analysis
        grid_resolution = 10  # Small grid for detailed analysis
        grid_width = int((max_x - min_x) / grid_resolution) + 1
        grid_length = int((max_y - min_y) / grid_resolution) + 1

        # Create occupation grid
        occupation_grid = np.zeros((grid_width, grid_length))

        # Fill in grid based on item positions
        for _, item in layer_items.iterrows():
            # Calculate grid coordinates
            grid_x_start = max(0, int((item["x"] - min_x) / grid_resolution))
            grid_y_start = max(0, int((item["y"] - min_y) / grid_resolution))
            grid_x_end = min(
                grid_width, int((item["x"] + item["width"] - min_x) / grid_resolution) + 1
            )
            grid_y_end = min(
                grid_length, int((item["y"] + item["length"] - min_y) / grid_resolution) + 1
            )

            # Mark as occupied
            for i in range(grid_x_start, grid_x_end):
                for j in range(grid_y_start, grid_y_end):
                    if i < grid_width and j < grid_length:
                        occupation_grid[i, j] = 1

        # Invert grid to get voids
        void_grid = 1 - occupation_grid

        # Use connected component analysis to find contiguous voids
        from scipy import ndimage

        labeled_voids, num_voids = ndimage.label(void_grid)

        if num_voids == 0:
            logger.info(f"No significant voids found in layer {current_layer}")
            return filled_df

        logger.info(f"Found {num_voids} distinct voids in layer {current_layer}")

        # Get sizes of each void
        void_sizes = []
        for void_id in range(1, num_voids + 1):
            void_size = np.sum(labeled_voids == void_id)
            void_sizes.append((void_id, void_size))

        # Sort voids by size (largest first)
        void_sizes.sort(key=lambda x: x[1], reverse=True)

        # Process large voids to determine if they can be filled
        for void_id, void_size in void_sizes:
            if void_size < 5:  # Skip very small voids
                continue

            # Get void bounds in grid coordinates
            void_mask = labeled_voids == void_id
            void_indices = np.where(void_mask)
            min_i, max_i = min(void_indices[0]), max(void_indices[0])
            min_j, max_j = min(void_indices[1]), max(void_indices[1])

            # Convert to actual coordinates
            void_min_x = min_i * grid_resolution + min_x
            void_min_y = min_j * grid_resolution + min_y
            void_max_x = (max_i + 1) * grid_resolution + min_x
            void_max_y = (max_j + 1) * grid_resolution + min_y

            # Calculate void dimensions
            void_width = void_max_x - void_min_x
            void_length = void_max_y - void_min_y

            logger.info(
                f"Void {void_id}: Size={void_size}, Dims=({void_width}x{void_length}), "
                + f"Pos=({void_min_x}, {void_min_y})"
            )

            # Check if this void is worth filling
            # Currently just logging information, actual filling would be added here

        return filled_df

    def _analyze_and_optimize_overall_packing(self, df):
        """Analyze the overall bin packing to identify and fix stability issues."""
        if df.empty:
            return df

        logger.info("Performing overall packing analysis and optimization")

        # Create a working copy
        optimized_df = df.copy()

        # Calculate overall packing density
        total_item_volume = sum(
            row.width * row.length * row.height for _, row in optimized_df.iterrows()
        )
        pallet_volume = self.pallet_dims.width * self.pallet_dims.length * self.pallet_dims.height
        overall_density = total_item_volume / pallet_volume if pallet_volume > 0 else 0

        logger.info(f"Overall packing density before optimization: {overall_density*100:.1f}%")

        # Calculate center of gravity using actual weights
        # Check if weight column exists, otherwise use default weight
        if "weight" in optimized_df.columns:
            total_weight = optimized_df["weight"].sum()
            weighted_x = sum(
                (row.x + row.width / 2) * row.weight for _, row in optimized_df.iterrows()
            )
            weighted_y = sum(
                (row.y + row.length / 2) * row.weight for _, row in optimized_df.iterrows()
            )
            weighted_z = sum(
                (row.z + row.height / 2) * row.weight for _, row in optimized_df.iterrows()
            )
        else:
            # Fallback to unit weight if weight column not available
            logger.warning(
                "Weight column not found in dataframe. Using uniform weight of 1.0 for all items."
            )
            total_weight = len(optimized_df)
            weighted_x = sum((row.x + row.width / 2) for _, row in optimized_df.iterrows())
            weighted_y = sum((row.y + row.length / 2) for _, row in optimized_df.iterrows())
            weighted_z = sum((row.z + row.height / 2) for _, row in optimized_df.iterrows())

        cog_x = weighted_x / total_weight if total_weight > 0 else 0
        cog_y = weighted_y / total_weight if total_weight > 0 else 0
        cog_z = weighted_z / total_weight if total_weight > 0 else 0

        # Calculate ideal center (middle of pallet)
        ideal_x = self.pallet_dims.width / 2
        ideal_y = self.pallet_dims.length / 2

        # Calculate distance from ideal
        cog_offset = np.sqrt((cog_x - ideal_x) ** 2 + (cog_y - ideal_y) ** 2)

        logger.info(f"Center of gravity: ({cog_x:.1f}, {cog_y:.1f}, {cog_z:.1f})")
        logger.info(f"Distance from ideal center: {cog_offset:.1f}")

        # If density is too low or COG is far from center, try optimizing
        if overall_density < 0.7 or cog_offset > self.pallet_dims.width * 0.2:
            logger.info("Bin packing needs improvement - attempting additional optimization")

            # Focus on specific issues - currently just logging, would add specific fixes
            if cog_offset > self.pallet_dims.width * 0.2:
                logger.info("Center of gravity is significantly off-center - would rebalance")

            if overall_density < 0.7:
                logger.info("Density is suboptimal - would attempt compaction")

        # Calculate overall stability score
        item_list = optimized_df.to_dict("records")
        stability_scores = []

        for item in item_list:
            if abs(item["z"]) < 0.1:  # Ground level items
                stability_scores.append(1.0)  # Perfect stability
            else:
                # Calculate support
                support_percentage, _ = self._calculate_support(
                    item["x"],
                    item["y"],
                    item["z"],
                    item["width"],
                    item["length"],
                    [i for i in item_list if i["item"] != item["item"]],
                    0,
                )

                # Calculate side contact
                side_contact_data = self._calculate_enhanced_side_contact(
                    item["x"],
                    item["y"],
                    item["z"],
                    item["width"],
                    item["length"],
                    item["height"],
                    [i for i in item_list if i["item"] != item["item"]],
                )

                # Combined stability score (support + side contact)
                stability = support_percentage * 0.7 + side_contact_data["contact_area_ratio"] * 0.3
                stability_scores.append(stability)

        average_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0
        logger.info(f"Average stability score: {average_stability*100:.1f}%")

        return optimized_df

    def _compact_layer_items(self, df, current_layer, force_clustering=True):
        """Post-process layer items by pushing them as close as possible to one another."""
        # Create a working copy
        compacted_df = df.copy()

        # Get all items in the current layer that have been processed
        layer_mask = (compacted_df["layer"] == current_layer) & (compacted_df["processed"] == True)
        layer_items = compacted_df[layer_mask].copy()

        if layer_items.empty:
            return compacted_df

        logger.info(f"Compacting {len(layer_items)} items in layer {current_layer}")

        # Get all items from previous layers (already placed and fixed)
        previous_items = compacted_df[
            compacted_df["processed"] & (compacted_df["layer"] < current_layer)
        ].copy()
        previous_items_list = previous_items.to_dict("records") if not previous_items.empty else []

        # Direction vectors for compaction (order matters - try origin-ward movements first)
        directions = [
            (-1, 0),  # Left (negative x)
            (0, -1),  # Down (negative y)
            (-1, -1),  # Diagonal toward origin
            (1, 0),  # Right (positive x)
            (0, 1),  # Up (positive y)
            (1, 1),  # Diagonal away from origin
            (1, -1),  # Diagonal right-down
            (-1, 1),  # Diagonal left-up
        ]

        # Sort items by distance from origin for better compaction results
        layer_items["distance_from_origin"] = layer_items["x"] + layer_items["y"]
        layer_items = layer_items.sort_values("distance_from_origin")

        # Normal compaction - no aggressive clustering

        # Process each item in the current layer
        for idx in layer_items.index:
            item = layer_items.loc[idx]
            item_id = item["item"]

            # Original position
            original_x, original_y, z = item["x"], item["y"], item["z"]
            width, length, height = item["width"], item["length"], item["height"]

            # Get all other items (excluding the current item)
            other_items = compacted_df[
                compacted_df["processed"] & (compacted_df["item"] != item_id)
            ].copy()
            other_items_list = other_items.to_dict("records")

            # Best position found so far (start with current position)
            best_x, best_y = original_x, original_y
            best_distance_to_origin = original_x + original_y

            # For force_clustering mode, we also consider side contact maximization
            best_side_contact = 0

            # Flag to track if position changed
            position_changed = False

            # Multi-step compaction - keep trying until no more movement is possible
            max_iterations = 10  # Normal compaction uses fewer iterations
            current_iteration = 0

            # Normal compaction - no aggressive clustering

            # Standard compaction process
            while current_iteration < max_iterations:
                current_iteration += 1
                moved_in_iteration = False

                # Try each direction
                for dx, dy in directions:
                    # Start with normal step size, then reduce it
                    step_sizes = [25, 10, 5, 1]

                    for step_size in step_sizes:
                        # Proposed new position
                        new_x = best_x + dx * step_size
                        new_y = best_y + dy * step_size

                        # Skip if outside pallet bounds
                        if not self._is_within_pallet_bounds(
                            new_x, new_y, z, width, length, height
                        ):
                            continue

                        # Skip if overlaps with other items
                        if self._check_overlap(
                            new_x, new_y, z, width, length, height, other_items_list
                        ):
                            continue

                        # Calculate support at new position
                        support_percentage, is_supported = self._calculate_support(
                            new_x,
                            new_y,
                            z,
                            width,
                            length,
                            other_items_list,
                            0.75,  # Maintain good support
                        )

                        # If ground level, support is always adequate
                        if abs(z) < 0.1:
                            is_supported = True

                        # Only accept if adequately supported
                        if not is_supported:
                            continue

                        # Standard compaction - just minimize distance to origin
                        new_distance_to_origin = new_x + new_y

                        if new_distance_to_origin < best_distance_to_origin:
                            best_x, best_y = new_x, new_y
                            best_distance_to_origin = new_distance_to_origin
                            moved_in_iteration = True
                            position_changed = True

                            # Break out of step size loop if we found a better position
                            break

                    # If we moved in this direction, try another direction immediately
                    if moved_in_iteration:
                        break

                # If no movement occurred in this iteration, we're done
                if not moved_in_iteration:
                    break

            # If position changed, update the dataframe
            if position_changed:
                # Update positions in both DataFrames
                compacted_df.loc[idx, "x"] = best_x
                compacted_df.loc[idx, "y"] = best_y
                layer_items.loc[idx, "x"] = best_x
                layer_items.loc[idx, "y"] = best_y

                logger.info(
                    f"Compacted item {item_id} from ({original_x}, {original_y}) to ({best_x}, {best_y})"
                )

                # Update this item in the other_items_list for subsequent item processing
                for i, other_item in enumerate(other_items_list):
                    if other_item["item"] == item_id:
                        other_items_list[i]["x"] = best_x
                        other_items_list[i]["y"] = best_y
                        break

        # Normal compaction - no void analysis needed

        # Drop temporary columns
        columns_to_drop = ["distance_from_origin"]

        # Calculate compaction efficiency metrics
        moved_items = sum(
            1
            for idx in layer_items.index
            if (
                layer_items.loc[idx, "x"] != df.loc[idx, "x"]
                or layer_items.loc[idx, "y"] != df.loc[idx, "y"]
            )
        )

        if moved_items > 0:
            logger.info(
                f"Layer {current_layer} compaction: {moved_items}/{len(layer_items)} items repositioned"
            )

            # Calculate bounding box before and after
            original_min_x = df.loc[layer_mask, "x"].min()
            original_min_y = df.loc[layer_mask, "y"].min()
            original_max_x = df.loc[layer_mask, "x"].max() + df.loc[layer_mask, "width"].max()
            original_max_y = df.loc[layer_mask, "y"].max() + df.loc[layer_mask, "length"].max()

            new_min_x = compacted_df.loc[layer_mask, "x"].min()
            new_min_y = compacted_df.loc[layer_mask, "y"].min()
            new_max_x = (
                compacted_df.loc[layer_mask, "x"].max()
                + compacted_df.loc[layer_mask, "width"].max()
            )
            new_max_y = (
                compacted_df.loc[layer_mask, "y"].max()
                + compacted_df.loc[layer_mask, "length"].max()
            )

            # Calculate area reduction
            original_area = (original_max_x - original_min_x) * (original_max_y - original_min_y)
            new_area = (new_max_x - new_min_x) * (new_max_y - new_min_y)

            area_reduction = (original_area - new_area) / original_area if original_area > 0 else 0
            logger.info(f"Layer {current_layer} area reduction: {area_reduction*100:.1f}%")

        # Clean up the DataFrame before returning
        for col in columns_to_drop:
            if col in compacted_df.columns:
                compacted_df = compacted_df.drop(columns=[col])

        return compacted_df

    # New method to find a position at an existing z-level
    def _find_position_at_existing_z(self, item, processed_items, z_level):
        """Find a valid position for an item at an existing z-level."""
        # Extract item dimensions
        width, length, height = item["width"], item["length"], item["height"]

        # Convert processed items to list
        items_list = processed_items.to_dict("records")

        # Check if there's enough vertical space
        if z_level + height > self.pallet_dims.height:
            return None

        # Minimum support required
        min_support_threshold = 0.75  # Require good support when reusing z-levels

        # Get all items at this z-level for better placement optimization
        items_at_level = [item for item in items_list if abs(item["z"] - z_level) < 0.1]

        # If no items at this level, try standard grid-based placement
        if not items_at_level:
            # Use a grid-based approach to find a position
            grid_resolution = 10
            grid_width = int(self.pallet_dims.width / grid_resolution) + 1
            grid_length = int(self.pallet_dims.length / grid_resolution) + 1

            # Create grid for this z-level (if items exist below)
            occupation_grid = [[False for _ in range(grid_length)] for _ in range(grid_width)]

            # Try each potential position
            for grid_x in range(grid_width - int(width / grid_resolution) - 1):
                for grid_y in range(grid_length - int(length / grid_resolution) - 1):
                    # Convert to actual coordinates
                    x = grid_x * grid_resolution
                    y = grid_y * grid_resolution
                    z = z_level

                    # Skip if outside pallet bounds
                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    # Skip if overlaps with other items
                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    # Check for proper support
                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, min_support_threshold
                    )

                    if is_supported:
                        return {"x": x, "y": y, "z": z, "support": support_percentage}
        else:
            # Try placing adjacent to existing items at this z-level
            for level_item in items_at_level:
                # Try adjacent positions to maximize side contact
                adjacent_positions = [
                    # Direct side contact positions
                    {"x": level_item["x"] + level_item["width"], "y": level_item["y"]},
                    {"x": level_item["x"] - width, "y": level_item["y"]},
                    {"x": level_item["x"], "y": level_item["y"] + level_item["length"]},
                    {"x": level_item["x"], "y": level_item["y"] - length},
                    # Corner positions (potentially less side contact but still valuable)
                    {
                        "x": level_item["x"] + level_item["width"],
                        "y": level_item["y"] + level_item["length"],
                    },
                    {"x": level_item["x"] - width, "y": level_item["y"] - length},
                    {"x": level_item["x"] + level_item["width"], "y": level_item["y"] - length},
                    {"x": level_item["x"] - width, "y": level_item["y"] + level_item["length"]},
                ]

                # Try each position
                for pos in adjacent_positions:
                    x, y = pos["x"], pos["y"]
                    z = z_level

                    # Skip if outside pallet bounds
                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    # Skip if overlaps with other items
                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    # Check for proper support
                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, min_support_threshold
                    )

                    if is_supported:
                        # Calculate side contact (higher is better)
                        side_contact_score = self._calculate_side_contact(
                            x, y, z, width, length, height, items_list
                        )

                        return {
                            "x": x,
                            "y": y,
                            "z": z,
                            "support": support_percentage,
                            "side_contact": side_contact_score,
                        }

        # If no position found using adjacent placement, try a more exhaustive grid search
        for x in range(0, int(self.pallet_dims.width - width + 1), 50):
            for y in range(0, int(self.pallet_dims.length - length + 1), 50):
                # Skip if outside pallet bounds
                if not self._is_within_pallet_bounds(x, y, z_level, width, length, height):
                    continue

                # Skip if overlaps with other items
                if self._check_overlap(x, y, z_level, width, length, height, items_list):
                    continue

                # Check for proper support
                support_percentage, is_supported = self._calculate_support(
                    x, y, z_level, width, length, items_list, min_support_threshold
                )

                if is_supported:
                    return {"x": x, "y": y, "z": z_level, "support": support_percentage}

        # No valid position found at this z-level
        return None

    def _optimize_layer0_items(self, layer0_items):
        """Optimize the placement of layer 0 items while respecting original z-coordinates."""
        # Sort items by base area in descending order
        layer0_items["base_area"] = layer0_items["width"] * layer0_items["length"]
        layer0_items = layer0_items.sort_values("base_area", ascending=False)

        # Place items using a skyline placement strategy
        placed_items = []

        # Initialize grid system
        grid_resolution = 10
        grid_width = int(self.pallet_dims.width / grid_resolution) + 1
        grid_length = int(self.pallet_dims.length / grid_resolution) + 1
        occupation_grids = {}  # Dictionary to store grids for different z-levels

        for idx in layer0_items.index:
            item = layer0_items.loc[idx]
            item_id = item["item"]

            # First check if current position is valid
            current_x, current_y, current_z = item["x"], item["y"], item["z"]
            current_position_valid = True

            # Check pallet boundaries
            if not self._is_within_pallet_bounds(
                current_x, current_y, current_z, item["width"], item["length"], item["height"]
            ):
                current_position_valid = False

            # Check for overlaps with placed items
            if current_position_valid and self._check_overlap(
                current_x,
                current_y,
                current_z,
                item["width"],
                item["length"],
                item["height"],
                placed_items,
            ):
                current_position_valid = False

            # If current position is valid, keep it
            if current_position_valid:
                self._update_occupation_grid(
                    occupation_grids,
                    current_x,
                    current_y,
                    current_z,
                    item["width"],
                    item["length"],
                    grid_resolution,
                    grid_width,
                    grid_length,
                )

                placed_items.append(
                    self._create_item_dict(
                        item_id,
                        current_x,
                        current_y,
                        current_z,
                        item["width"],
                        item["length"],
                        item["height"],
                    )
                )

                logger.info(
                    f"Kept layer0 item {item_id} at original position ({current_x}, {current_y}, {current_z})"
                )
                continue

            # If original position invalid, find a new position at same z-level
            z_level = current_z  # Maintain the original z-coordinate

            # Get or create grid for this z-level
            if z_level not in occupation_grids:
                occupation_grids[z_level] = [
                    [False for _ in range(grid_length)] for _ in range(grid_width)
                ]

            # Find best position using grid-based approach
            position = self._find_grid_position(
                item,
                z_level,
                occupation_grids[z_level],
                grid_resolution,
                grid_width,
                grid_length,
                placed_items,
            )

            if position:
                # Update item position
                layer0_items.loc[idx, "x"] = position["x"]
                layer0_items.loc[idx, "y"] = position["y"]
                # z-coordinate stays the same

                # Update grid and placed items
                self._update_occupation_grid(
                    occupation_grids,
                    position["x"],
                    position["y"],
                    z_level,
                    item["width"],
                    item["length"],
                    grid_resolution,
                    grid_width,
                    grid_length,
                )

                placed_items.append(
                    self._create_item_dict(
                        item_id,
                        position["x"],
                        position["y"],
                        z_level,
                        item["width"],
                        item["length"],
                        item["height"],
                    )
                )

                logger.info(
                    f"Placed layer0 item {item_id} at ({position['x']}, {position['y']}, {z_level})"
                )
            else:
                # Try fallback placement methods
                position = self._try_fallback_layer0_placement(item, z_level, placed_items)

                if position:
                    layer0_items.loc[idx, "x"] = position["x"]
                    layer0_items.loc[idx, "y"] = position["y"]
                    layer0_items.loc[idx, "z"] = position["z"]  # May change if using alternative z

                    placed_items.append(
                        self._create_item_dict(
                            item_id,
                            position["x"],
                            position["y"],
                            position["z"],
                            item["width"],
                            item["length"],
                            item["height"],
                        )
                    )

                    placement_type = (
                        "at origin"
                        if position.get("at_origin")
                        else (
                            "at boundary"
                            if position.get("at_boundary")
                            else (
                                "with fine grid"
                                if position.get("fine_grid")
                                else "at alternative z"
                            )
                        )
                    )

                    logger.info(
                        f"Placed layer0 item {item_id} {placement_type} ({position['x']}, {position['y']}, {position['z']})"
                    )
                else:
                    logger.warning(f"Could not place layer0 item {item_id} - will try in next bin")

        # Drop temporary columns
        if "base_area" in layer0_items.columns:
            layer0_items = layer0_items.drop(columns=["base_area"])

        return layer0_items

    def _create_item_dict(self, item_id, x, y, z, width, length, height, **kwargs):
        """Helper to create a standardized item dictionary."""
        item_dict = {
            "item": item_id,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "length": length,
            "height": height,
        }
        item_dict.update(kwargs)  # Add any additional fields
        return item_dict

    def _is_within_pallet_bounds(self, x, y, z, width, length, height):
        """Check if an item position is within pallet boundaries."""
        return (
            x >= 0
            and y >= 0
            and z >= 0
            and x + width <= self.pallet_dims.width
            and y + length <= self.pallet_dims.length
            and z + height <= self.pallet_dims.height
        )

    def _check_overlap(self, x, y, z, width, length, height, items_list):
        """Check if a position overlaps with any items in the list."""
        for item in items_list:
            x_overlap = max(0, min(x + width, item["x"] + item["width"]) - max(x, item["x"]))
            y_overlap = max(0, min(y + length, item["y"] + item["length"]) - max(y, item["y"]))
            z_overlap = max(0, min(z + height, item["z"] + item["height"]) - max(z, item["z"]))

            if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
                return True  # Overlap found

        return False  # No overlap

    def _update_occupation_grid(
        self, occupation_grids, x, y, z, width, length, grid_resolution, grid_width, grid_length
    ):
        """Update the occupation grid for a given position."""
        if z not in occupation_grids:
            occupation_grids[z] = [[False for _ in range(grid_length)] for _ in range(grid_width)]

        grid_x_start = max(0, int(x / grid_resolution))
        grid_y_start = max(0, int(y / grid_resolution))
        grid_x_end = min(grid_width, int((x + width) / grid_resolution) + 1)
        grid_y_end = min(grid_length, int((y + length) / grid_resolution) + 1)

        for grid_x in range(grid_x_start, grid_x_end):
            for grid_y in range(grid_y_start, grid_y_end):
                if grid_x < grid_width and grid_y < grid_length:
                    occupation_grids[z][grid_x][grid_y] = True

    def _find_grid_position(
        self, item, z_level, occupation_grid, grid_resolution, grid_width, grid_length, placed_items
    ):
        """Find best position in grid with edge prioritization."""
        # Convert item dimensions to grid units
        item_grid_width = int(item["width"] / grid_resolution) + 1
        item_grid_length = int(item["length"] / grid_resolution) + 1

        best_position = None
        best_score = float("inf")

        # Try each potential position
        for grid_x in range(grid_width - item_grid_width + 1):
            for grid_y in range(grid_length - item_grid_length + 1):
                # Check grid position validity
                valid = True
                for dx in range(item_grid_width):
                    for dy in range(item_grid_length):
                        if grid_x + dx < grid_width and grid_y + dy < grid_length:
                            if occupation_grid[grid_x + dx][grid_y + dy]:
                                valid = False
                                break
                    if not valid:
                        break

                if not valid:
                    continue

                # Convert to actual coordinates
                x = grid_x * grid_resolution
                y = grid_y * grid_resolution

                # Check for 3D overlaps with items at other z-levels
                if self._check_overlap(
                    x, y, z_level, item["width"], item["length"], item["height"], placed_items
                ):
                    continue

                # Calculate score based on edge proximity and distance from origin
                is_left_edge = grid_x == 0
                is_right_edge = grid_x + item_grid_width == grid_width
                is_bottom_edge = grid_y == 0
                is_top_edge = grid_y + item_grid_length == grid_length

                num_edges = is_left_edge + is_right_edge + is_bottom_edge + is_top_edge
                origin_distance = x + y

                score = origin_distance - (num_edges * 100)

                if score < best_score:
                    best_score = score
                    best_position = {"x": x, "y": y}

        return best_position

    def _try_fallback_layer0_placement(self, item, z_level, placed_items):
        """Try different fallback placement strategies for layer 0 items."""
        # Try origin placement if no items yet
        if not placed_items:
            return {"x": 0, "y": 0, "z": z_level, "at_origin": True}

        # Try pallet boundary placement
        x, y, found = self._find_boundary_position(
            item["width"], item["length"], z_level, item["height"], placed_items
        )

        if found:
            return {"x": x, "y": y, "z": z_level, "at_boundary": True}

        # Try fine grid search
        x, y, found = self._find_fine_grid_position(
            item["width"], item["length"], z_level, item["height"], placed_items
        )

        if found:
            return {"x": x, "y": y, "z": z_level, "fine_grid": True}

        # Try alternative z-level
        return self._find_alternative_z_position(
            item["width"], item["length"], item["height"], placed_items
        )

    def _calculate_support(self, x, y, z, width, length, items, min_support_threshold=0):
        """Calculate support percentage for an item at the given position."""
        base_area = width * length

        # If on ground level, 100% support
        if abs(z) < 0.1:
            return 1.0, True

        # Find items that could provide support
        support_items = [item for item in items if item["z"] + item["height"] <= z]
        supported_area = 0

        for support_item in support_items:
            # Check if there's direct support (tops touch bottoms)
            if abs(support_item["z"] + support_item["height"] - z) < 0.1:
                # Calculate overlap area
                x_overlap = max(
                    0,
                    min(x + width, support_item["x"] + support_item["width"])
                    - max(x, support_item["x"]),
                )
                y_overlap = max(
                    0,
                    min(y + length, support_item["y"] + support_item["length"])
                    - max(y, support_item["y"]),
                )

                if x_overlap > 0 and y_overlap > 0:
                    supported_area += x_overlap * y_overlap

        # Calculate support percentage
        support_percentage = supported_area / base_area if base_area > 0 else 0
        is_supported = support_percentage >= min_support_threshold

        return support_percentage, is_supported

    def _find_boundary_position(self, width, length, z, height, placed_items):
        """Try to find a position along the pallet boundary."""
        # The edges to try in order
        edges = [
            # x=0 edge with increasing y
            (
                lambda i: 0,
                lambda i: i * 10,
                lambda i: i < int(self.pallet_dims.length - length + 1) / 10,
            ),
            # y=0 edge with increasing x
            (
                lambda i: i * 10,
                lambda i: 0,
                lambda i: i < int(self.pallet_dims.width - width + 1) / 10,
            ),
            # right edge
            (
                lambda i: max(0, self.pallet_dims.width - width),
                lambda i: i * 10,
                lambda i: i < int(self.pallet_dims.length - length + 1) / 10,
            ),
            # top edge
            (
                lambda i: i * 10,
                lambda i: max(0, self.pallet_dims.length - length),
                lambda i: i < int(self.pallet_dims.width - width + 1) / 10,
            ),
        ]

        # Try each edge
        for get_x, get_y, condition in edges:
            i = 0
            while condition(i):
                x, y = get_x(i), get_y(i)

                # Check for overlap
                if not self._check_overlap(x, y, z, width, length, height, placed_items):
                    return x, y, True

                i += 1

        return 0, 0, False

    def _find_fine_grid_position(self, width, length, z, height, placed_items):
        """Try a fine grid search across the entire pallet at a specific z-level."""
        step_size = 50  # Finer grid step

        for x in range(0, int(self.pallet_dims.width - width + 1), step_size):
            for y in range(0, int(self.pallet_dims.length - length + 1), step_size):
                if not self._check_overlap(x, y, z, width, length, height, placed_items):
                    return x, y, True

        return 0, 0, False

    def _find_alternative_z_position(self, width, length, height, placed_items):
        """Find an alternative z-level position for an item."""
        z_steps = 500

        for z in range(0, int(self.pallet_dims.height - height + 1), z_steps):
            for x in range(0, int(self.pallet_dims.width - width + 1), 5):
                for y in range(0, int(self.pallet_dims.length - length + 1), 5):
                    if not self._check_overlap(x, y, z, width, length, height, placed_items):
                        return {"x": x, "y": y, "z": z}

        return None

    def _find_interlocking_position(self, item, processed_items):
        """Find positions where the item interlocks with existing items by fitting."""
        # Extract item dimensions
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        # Get all unique z-levels
        z_levels = sorted(set(item["z"] for item in items_list)) if items_list else [0]

        # Track best position with score
        best_position = None
        best_score = float("-inf")  # Higher is better for interlocking

        # For each z-level, find potential interlocking positions
        for z_level in z_levels:
            # Skip if this level would exceed height
            if z_level + height > self.pallet_dims.height:
                continue

            # Get items at this z-level
            items_at_level = [item for item in items_list if abs(item["z"] - z_level) < 0.1]

            # If fewer than 2 items at this level, not enough for interlocking
            if len(items_at_level) < 2:
                continue

            # Find potential concavities (positions with multiple adjacent items)
            for step_x in range(0, int(self.pallet_dims.width - width + 1), 10):
                for step_y in range(0, int(self.pallet_dims.length - length + 1), 10):
                    x, y, z = step_x, step_y, z_level

                    # Skip if outside pallet
                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    # Skip if overlaps
                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    # Calculate support
                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, 0.7  # Higher threshold for interlocking
                    )

                    if not is_supported:
                        continue

                    # Count number of different items providing side contact
                    side_contact_score = self._calculate_side_contact(
                        x, y, z, width, length, height, items_list
                    )

                    # Count number of distinct items with side contact
                    side_contact_items = self._count_side_contact_items(
                        x, y, z, width, length, height, items_list
                    )

                    # Position is good for interlocking if multiple items provide side contact
                    if side_contact_items >= 2:
                        # Calculate interlocking score
                        # Heavily favor positions with contact from multiple items
                        interlocking_score = (
                            side_contact_score * 100  # Base side contact percentage
                            + side_contact_items * 500  # Bonus for each contact item
                            + (1 - (x + y) / (self.pallet_dims.width + self.pallet_dims.length))
                            * 50  # Prefer closer to origin
                        )

                        # If this is better than current best, update
                        if interlocking_score > best_score:
                            best_score = interlocking_score
                            best_position = {
                                "x": x,
                                "y": y,
                                "z": z,
                                "support": support_percentage,
                                "side_contact": side_contact_score,
                                "contact_items": side_contact_items,
                                "score": interlocking_score,
                            }

        return best_position

    def _count_side_contact_items(self, x, y, z, width, length, height, items_list):
        """Count the number of distinct items providing side contact."""
        contact_items = set()

        # Check each side of the item against other items
        for other_item in items_list:
            # Left side (x-axis)
            if abs(other_item["x"] + other_item["width"] - x) < 0.1:
                # Check for overlap in y-z plane
                y_overlap = max(
                    0,
                    min(y + length, other_item["y"] + other_item["length"])
                    - max(y, other_item["y"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                if y_overlap > 0 and z_overlap > 0:
                    contact_items.add(other_item["item"])

            # Right side (x-axis)
            if abs(other_item["x"] - (x + width)) < 0.1:
                # Check for overlap in y-z plane
                y_overlap = max(
                    0,
                    min(y + length, other_item["y"] + other_item["length"])
                    - max(y, other_item["y"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                if y_overlap > 0 and z_overlap > 0:
                    contact_items.add(other_item["item"])

            # Front side (y-axis)
            if abs(other_item["y"] + other_item["length"] - y) < 0.1:
                # Check for overlap in x-z plane
                x_overlap = max(
                    0,
                    min(x + width, other_item["x"] + other_item["width"]) - max(x, other_item["x"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                if x_overlap > 0 and z_overlap > 0:
                    contact_items.add(other_item["item"])

            # Back side (y-axis)
            if abs(other_item["y"] - (y + length)) < 0.1:
                # Check for overlap in x-z plane
                x_overlap = max(
                    0,
                    min(x + width, other_item["x"] + other_item["width"]) - max(x, other_item["x"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                if x_overlap > 0 and z_overlap > 0:
                    contact_items.add(other_item["item"])

        return len(contact_items)

    # Enhance the _find_optimal_position method to prioritize side support
    def _find_optimal_position(self, item, processed_items, current_layer, current_layer_height):
        """Find the optimal position for an item that maximizes horizontal coverage,."""
        items_list = processed_items.to_dict("records")
        min_support_threshold = 0.85

        def try_position(width, length, rotated):
            potential_positions = []
            z_levels = sorted(set(i["z"] for i in items_list)) if items_list else [0]
            if current_layer_height > 0 and current_layer_height not in z_levels:
                z_levels.append(current_layer_height)
            for z_level in z_levels:
                if z_level + item["height"] > self.pallet_dims.height:
                    continue
                items_at_level = [i for i in items_list if i["z"] == z_level]
                for level_item in items_at_level:
                    adjacent_positions = [
                        {"x": level_item["x"] + level_item["width"], "y": level_item["y"]},
                        {"x": level_item["x"] - width, "y": level_item["y"]},
                        {"x": level_item["x"], "y": level_item["y"] + level_item["length"]},
                        {"x": level_item["x"], "y": level_item["y"] - length},
                        {
                            "x": level_item["x"] + level_item["width"],
                            "y": level_item["y"] + level_item["length"],
                        },
                        {"x": level_item["x"] - width, "y": level_item["y"] - length},
                        {"x": level_item["x"] + level_item["width"], "y": level_item["y"] - length},
                        {"x": level_item["x"] - width, "y": level_item["y"] + level_item["length"]},
                    ]
                    for pos in adjacent_positions:
                        x, y, z = pos["x"], pos["y"], z_level
                        if not self._is_within_pallet_bounds(
                            x, y, z, width, length, item["height"]
                        ):
                            continue
                        if self._check_overlap(x, y, z, width, length, item["height"], items_list):
                            continue
                        support_percentage, is_supported = self._calculate_support(
                            x, y, z, width, length, items_list, min_support_threshold
                        )
                        if is_supported:
                            # Calculate comprehensive horizontal spread score
                            spread_data = self._calculate_horizontal_spread_score(
                                x, y, z, width, length, item["height"], items_list
                            )

                            side_contact_score = self._calculate_side_contact(
                                x, y, z, width, length, item["height"], items_list
                            )
                            cog_contribution = self._calculate_cog_contribution(
                                x, y, z, width, length, item["height"]
                            )
                            hw_ratio_contribution = self._calculate_height_width_contribution(
                                x, y, z, width, length, item["height"]
                            )

                            # Enhanced scoring with horizontal spread maximization
                            height_score = (z**2) * 1000
                            support_score = (1 - support_percentage) * 100
                            spread_score = spread_data[
                                "spread_score"
                            ]  # Use comprehensive spread score

                            # Calculate final score with spread maximization priority
                            score = (
                                height_score
                                + support_score
                                - (spread_score * 2)  # Heavily weight horizontal spread
                                - (side_contact_score * 500)
                                - (cog_contribution * 100)
                                - (hw_ratio_contribution * 100)
                            )

                            potential_positions.append(
                                {
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "width": width,
                                    "length": length,
                                    "rotated": rotated,
                                    "support": support_percentage,
                                    "side_contact": side_contact_score,
                                    "spread_score": spread_score,
                                    "spread_data": spread_data,
                                    "score": score,
                                }
                            )

            # Also try grid-based positions for better spread coverage
            if not potential_positions:
                # Try strategic grid positions for better horizontal spread
                grid_step = 50
                for x in range(0, int(self.pallet_dims.width - width + 1), grid_step):
                    for y in range(0, int(self.pallet_dims.length - length + 1), grid_step):
                        z = z_levels[0] if z_levels else 0  # Try at the lowest z-level

                        if not self._is_within_pallet_bounds(
                            x, y, z, width, length, item["height"]
                        ):
                            continue
                        if self._check_overlap(x, y, z, width, length, item["height"], items_list):
                            continue

                        support_percentage, is_supported = self._calculate_support(
                            x, y, z, width, length, items_list, min_support_threshold
                        )
                        if is_supported:
                            # Calculate comprehensive horizontal spread score
                            spread_data = self._calculate_horizontal_spread_score(
                                x, y, z, width, length, item["height"], items_list
                            )

                            side_contact_score = self._calculate_side_contact(
                                x, y, z, width, length, item["height"], items_list
                            )

                            # Enhanced scoring for grid positions
                            height_score = (z**2) * 1000
                            support_score = (1 - support_percentage) * 100
                            spread_score = spread_data["spread_score"]

                            score = (
                                height_score
                                + support_score
                                - (spread_score * 2)  # Prioritize horizontal spread
                                - (
                                    side_contact_score * 300
                                )  # Reduced side contact weight for grid positions
                            )

                            potential_positions.append(
                                {
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "width": width,
                                    "length": length,
                                    "rotated": rotated,
                                    "support": support_percentage,
                                    "side_contact": side_contact_score,
                                    "spread_score": spread_score,
                                    "spread_data": spread_data,
                                    "score": score,
                                    "grid_position": True,
                                }
                            )

            if potential_positions:
                return min(potential_positions, key=lambda pos: pos["score"])
            return None

        return self._try_orientations(item, try_position)

    def _calculate_side_contact(self, x, y, z, width, length, height, items_list):
        """Calculate a score based on how much side contact an item would have."""
        total_side_area = 2 * (width * height + length * height)
        contact_area = 0

        # Check each side of the item against other items
        for other_item in items_list:
            # Left side (x-axis)
            if abs(other_item["x"] + other_item["width"] - x) < 0.1:
                # Other item's right side touches this item's left side
                y_overlap = max(
                    0,
                    min(y + length, other_item["y"] + other_item["length"])
                    - max(y, other_item["y"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                overlap_area = y_overlap * z_overlap

                # Boost score for large overlaps (>50% of face area)
                if overlap_area > 0.5 * length * height:
                    overlap_area *= 1.5  # 50% bonus for large overlaps

                contact_area += overlap_area

            # Right side (x-axis)
            if abs(other_item["x"] - (x + width)) < 0.1:
                # Similar calculations for right side with bonus for large overlaps
                y_overlap = max(
                    0,
                    min(y + length, other_item["y"] + other_item["length"])
                    - max(y, other_item["y"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                overlap_area = y_overlap * z_overlap

                if overlap_area > 0.5 * length * height:
                    overlap_area *= 1.5

                contact_area += overlap_area

            # Front side (y-axis)
            if abs(other_item["y"] + other_item["length"] - y) < 0.1:
                # Similar calculations for front side with bonus
                x_overlap = max(
                    0,
                    min(x + width, other_item["x"] + other_item["width"]) - max(x, other_item["x"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                overlap_area = x_overlap * z_overlap

                if overlap_area > 0.5 * width * height:
                    overlap_area *= 1.5

                contact_area += overlap_area

            # Back side (y-axis)
            if abs(other_item["y"] - (y + length)) < 0.1:
                # Similar calculations for back side with bonus
                x_overlap = max(
                    0,
                    min(x + width, other_item["x"] + other_item["width"]) - max(x, other_item["x"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                overlap_area = x_overlap * z_overlap

                if overlap_area > 0.5 * width * height:
                    overlap_area *= 1.5

                contact_area += overlap_area

        # Return score as ratio of contact area to total side area
        return contact_area / total_side_area if total_side_area > 0 else 0

    # New helper method to calculate center of gravity contribution
    def _calculate_cog_contribution(self, x, y, z, width, length, height):
        """Calculate how much an item at this position contributes to the."""
        # Calculate item center point
        item_center_x = x + width / 2
        item_center_y = y + length / 2

        # Calculate bin center point
        bin_center_x = self.pallet_dims.width / 2
        bin_center_y = self.pallet_dims.length / 2

        # Calculate normalized distance from center (0 = center, 1 = furthest corner)
        max_dist = np.sqrt(bin_center_x**2 + bin_center_y**2)

        # Calculate actual distance
        actual_dist = np.sqrt(
            (item_center_x - bin_center_x) ** 2 + (item_center_y - bin_center_y) ** 2
        )

        # Return score (0 = furthest, 1 = at center)
        normalized_dist = actual_dist / max_dist if max_dist > 0 else 0
        return 1 - normalized_dist

    # New helper method to calculate height-width ratio contribution
    def _calculate_height_width_contribution(self, x, y, z, width, length, height):
        """Calculate a score that rewards placing tall items (high height-to-base ratio)."""
        # Calculate height-to-base ratio
        base_area = width * length
        height_base_ratio = height / np.sqrt(base_area) if base_area > 0 else 0

        # Normalize to 0-1 range (assuming a reasonable max ratio of 5)
        normalized_ratio = min(1.0, height_base_ratio / 5.0)

        # Calculate distance from center (xy plane)
        bin_center_x = self.pallet_dims.width / 2
        bin_center_y = self.pallet_dims.length / 2
        item_center_x = x + width / 2
        item_center_y = y + length / 2

        # Calculate horizontal distance from center
        max_dist = np.sqrt(bin_center_x**2 + bin_center_y**2)
        actual_dist = np.sqrt(
            (item_center_x - bin_center_x) ** 2 + (item_center_y - bin_center_y) ** 2
        )

        # Normalize distance (0 = furthest, 1 = at center)
        normalized_dist = 1 - (actual_dist / max_dist) if max_dist > 0 else 1

        # Calculate normalized height (0 = bottom, 1 = top)
        normalized_height = z / self.pallet_dims.height if self.pallet_dims.height > 0 else 0

        # Tall items should be at the center and bottom
        # The more normalized_ratio, the more it matters
        score = normalized_ratio * (normalized_dist * 0.7 + (1 - normalized_height) * 0.3)

        return score

    def _overlaps(self, x1, y1, z1, w1, l1, h1, x2, y2, z2, w2, l2, h2):
        """Check if two 3D boxes overlap."""
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + l1, y2 + l2) - max(y1, y2))
        z_overlap = max(0, min(z1 + h1, z2 + h2) - max(z1, z2))

        return x_overlap > 0 and y_overlap > 0 and z_overlap > 0

    def _find_fallback_position(self, item, processed_items):
        """Find a fallback position with reduced support threshold."""
        # Reduced support threshold for fallback
        reduced_support_threshold = 0.6

        # Extract item dimensions and create list from DataFrame
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        # If no processed items, place at origin
        if not items_list:
            return {"x": 0, "y": 0, "z": 0}

        # Try positions on top of existing items with reduced support requirements
        for support_item in items_list:
            # Try multiple positions on this support item (corners)
            for x_offset in [0, support_item["width"] - width]:
                for y_offset in [0, support_item["length"] - length]:
                    x = support_item["x"] + x_offset
                    y = support_item["y"] + y_offset
                    z = support_item["z"] + support_item["height"]

                    # Skip invalid positions
                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    # Skip if overlaps
                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    # Calculate support percentage
                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, reduced_support_threshold
                    )

                    if is_supported:
                        logger.info(
                            f"Fallback: Found position with {support_percentage*100:.1f}% support"
                        )
                        return {"x": x, "y": y, "z": z, "support": support_percentage}

        return None

    def _find_alternative_position(self, item, processed_items):
        """Find any valid position as a last resort, checking for minimum support."""
        # Minimum support for alternative positions
        min_support_threshold = 0.3

        # Extract item dimensions and convert DataFrame to list
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records") if not processed_items.empty else []

        # Track used positions to prevent duplicates
        used_positions = set()

        # Try positions at different z-levels with grid search (original orientation)
        step_size = 50
        z_step = 250

        # Try from lowest to highest z-levels
        for z in range(0, int(self.pallet_dims.height - height + 1), z_step):
            for x in range(0, int(self.pallet_dims.width - width + 1), step_size):
                for y in range(0, int(self.pallet_dims.length - length + 1), step_size):
                    # Skip if position isn't valid
                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    # Skip if position already used
                    pos_key = (x, y, z, width, length, height)
                    if pos_key in used_positions:
                        continue

                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    # Check support
                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, min_support_threshold
                    )

                    # If on ground level or has sufficient support
                    if is_supported:
                        logger.info(
                            f"Found alternative position for item {item['item']} at ({x}, {y}, {z}) with {support_percentage*100:.1f}% support"
                        )
                        return {"x": x, "y": y, "z": z, "support": support_percentage}

        # Try with rotated orientation if dimensions are different
        if width != length:
            logger.info(
                f"Trying rotation for item {item['item']} (original: {width}x{length}x{height})"
            )

            for z in range(0, int(self.pallet_dims.height - height + 1), z_step):
                for x in range(0, int(self.pallet_dims.width - length + 1), step_size):
                    for y in range(0, int(self.pallet_dims.length - width + 1), step_size):
                        # Skip if position isn't valid with rotated dimensions
                        if not self._is_within_pallet_bounds(x, y, z, length, width, height):
                            continue

                        # Skip if position already used
                        pos_key = (x, y, z, length, width, height)
                        if pos_key in used_positions:
                            continue

                        # Check for overlaps with rotated dimensions
                        if self._check_overlap(x, y, z, length, width, height, items_list):
                            continue

                        # Calculate support with rotated dimensions
                        support_percentage, is_supported = self._calculate_support(
                            x, y, z, length, width, items_list, min_support_threshold
                        )

                        if is_supported:
                            logger.info(
                                f"Found alternative position for item {item['item']} with rotation at ({x}, {y}, {z}) with {support_percentage*100:.1f}% support"
                            )
                            return {
                                "x": x,
                                "y": y,
                                "z": z,
                                "width": length,  # Update dimensions after rotation
                                "length": width,
                                "rotated": True,
                                "support": support_percentage,
                            }

        # Enhanced rotation strategy: Try both 90-degree rotations if width != length
        if width != length:
            logger.info(f"Trying enhanced rotation strategies for item {item['item']}")

            # Define rotation configurations
            rotation_configs = [
                {"w": length, "l": width, "rotation": "90_cw"},  # 90 degrees clockwise
                {"w": width, "l": length, "rotation": "original"},  # Original (already tried above)
            ]

            # Try each rotation with finer grid search
            fine_step = 25  # Finer step for rotation attempts

            for config in rotation_configs:
                if config["rotation"] == "original":
                    continue  # Already tried above

                rot_width, rot_length = config["w"], config["l"]

                # Skip if rotated item won't fit in pallet at all
                if (
                    rot_width > self.pallet_dims.width
                    or rot_length > self.pallet_dims.length
                    or height > self.pallet_dims.height
                ):
                    continue

                for z in range(0, int(self.pallet_dims.height - height + 1), z_step):
                    for x in range(0, int(self.pallet_dims.width - rot_width + 1), fine_step):
                        for y in range(0, int(self.pallet_dims.length - rot_length + 1), fine_step):
                            # Check validity with rotated dimensions
                            if not self._is_within_pallet_bounds(
                                x, y, z, rot_width, rot_length, height
                            ):
                                continue

                            # Skip if position already used
                            pos_key = (x, y, z, rot_width, rot_length, height)
                            if pos_key in used_positions:
                                continue

                            if self._check_overlap(
                                x, y, z, rot_width, rot_length, height, items_list
                            ):
                                continue

                            # Calculate support
                            support_percentage, is_supported = self._calculate_support(
                                x, y, z, rot_width, rot_length, items_list, min_support_threshold
                            )

                            if is_supported:
                                logger.info(
                                    f"Found position for item {item['item']} with {config['rotation']} rotation at ({x}, {y}, {z}) with {support_percentage*100:.1f}% support"
                                )
                                return {
                                    "x": x,
                                    "y": y,
                                    "z": z,
                                    "width": rot_width,
                                    "length": rot_length,
                                    "rotated": True,
                                    "rotation_type": config["rotation"],
                                    "support": support_percentage,
                                }

        # Last resort: try with even less support, but ensure unique positions
        if items_list:
            reduced_threshold = 0.1  # Just 10% support as absolute minimum

            # Try only a few strategic positions with reduced support
            for support_item in items_list:
                if support_item["z"] + support_item["height"] + height <= self.pallet_dims.height:
                    # Try multiple positions around the support item to avoid duplicates
                    offset_positions = [
                        (support_item["x"], support_item["y"]),
                        (support_item["x"] + 10, support_item["y"]),
                        (support_item["x"], support_item["y"] + 10),
                        (support_item["x"] + 10, support_item["y"] + 10),
                        (support_item["x"] - 10, support_item["y"]),
                        (support_item["x"], support_item["y"] - 10),
                    ]

                    for x_offset, y_offset in offset_positions:
                        x, y = x_offset, y_offset
                    z = support_item["z"] + support_item["height"]

                    # Check if valid and not overlapping (original orientation)
                    if self._is_within_pallet_bounds(
                        x, y, z, width, length, height
                    ) and not self._check_overlap(
                        x,
                        y,
                        z,
                        width,
                        length,
                        height,
                        [item for item in items_list if item != support_item],
                    ):

                        # Skip if position already used
                        pos_key = (x, y, z, width, length, height)
                        if pos_key in used_positions:
                            continue

                        # Calculate support with minimal threshold
                        support_percentage, is_supported = self._calculate_support(
                            x, y, z, width, length, [support_item], reduced_threshold
                        )

                        if is_supported:
                            used_positions.add(pos_key)
                            logger.warning(
                                f"Last resort: Placed item {item['item']} at ({x}, {y}, {z}) with minimal support ({support_percentage*100:.1f}%)"
                            )
                            return {
                                "x": x,
                                "y": y,
                                "z": z,
                                "support": support_percentage,
                                "last_resort": True,
                            }

                    # Try rotated orientation as last resort
                    if width != length:
                        for x_offset, y_offset in offset_positions:
                            x, y = x_offset, y_offset
                            z = support_item["z"] + support_item["height"]

                            if self._is_within_pallet_bounds(
                                x, y, z, length, width, height
                            ) and not self._check_overlap(
                                x,
                                y,
                                z,
                                length,
                                width,
                                height,
                                [item for item in items_list if item != support_item],
                            ):

                                # Skip if position already used
                                pos_key = (x, y, z, length, width, height)
                                if pos_key in used_positions:
                                    continue

                                # Calculate support with minimal threshold and rotated dimensions
                                support_percentage, is_supported = self._calculate_support(
                                    x, y, z, length, width, [support_item], reduced_threshold
                                )

                                if is_supported:
                                    used_positions.add(pos_key)
                                    logger.warning(
                                        f"Last resort with rotation: Placed item {item['item']} at ({x}, {y}, {z}) with minimal support ({support_percentage*100:.1f}%)"
                                    )
                                    return {
                                        "x": x,
                                        "y": y,
                                        "z": z,
                                        "width": length,
                                        "length": width,
                                        "rotated": True,
                                        "support": support_percentage,
                                        "last_resort": True,
                                    }

        # If still no position found, return None
        logger.error(
            f"Could not find any valid position for item {item['item']} even with rotation"
        )
        return None

    def _validate_final_packing(self, input_df=None):
        """Perform a final validation of the entire bin packing solution."""

        # Create a clean copy of the final dataframe
        validated_df = self.df.copy() if input_df is None else input_df.copy()

        # Track validation issues
        validation_report = {
            "overlapping_items": [],
            "insufficient_support_items": [],
            "removed_items": [],
        }

        # Minimum support threshold for final validation
        final_min_support = 0.5  # 50% support required for final solution

        # Convert dataframe to list of dictionaries for easier processing
        items = validated_df.to_dict("records")

        # Check each item against all other items for overlaps and support
        for i, item in enumerate(items):
            item_id = item["item"]

            # Skip items that have already been removed
            if item_id in validation_report["removed_items"]:
                continue

            # Check for overlaps with all other items
            overlaps = []
            for j, other_item in enumerate(items):
                if i == j or other_item["item"] in validation_report["removed_items"]:
                    continue

                # Check for 3D overlap
                if self._overlaps(
                    item["x"],
                    item["y"],
                    item["z"],
                    item["width"],
                    item["length"],
                    item["height"],
                    other_item["x"],
                    other_item["y"],
                    other_item["z"],
                    other_item["width"],
                    other_item["length"],
                    other_item["height"],
                ):
                    overlaps.append(other_item["item"])

            # If overlaps detected, log and mark for removal
            if overlaps:
                logger.warning(f"Final validation: Item {item_id} overlaps with items {overlaps}")
                validation_report["overlapping_items"].append(
                    {"item": item_id, "overlaps_with": overlaps}
                )
                validation_report["removed_items"].append(item_id)
                continue  # Skip support check if already invalid

            # Check support only if not on ground level
            if item["z"] > 0.1:
                # Calculate support using other items that are not removed
                valid_supports = [
                    support_item
                    for support_item in items
                    if support_item["item"] not in validation_report["removed_items"]
                    and support_item["item"] != item_id
                ]

                support_percentage, is_supported = self._calculate_support(
                    item["x"],
                    item["y"],
                    item["z"],
                    item["width"],
                    item["length"],
                    valid_supports,
                    final_min_support,
                )

                if not is_supported:
                    logger.warning(
                        f"Final validation: Item {item_id} has insufficient support ({support_percentage*100:.1f}%)"
                    )
                    validation_report["insufficient_support_items"].append(
                        {"item": item_id, "support": support_percentage}
                    )
                    validation_report["removed_items"].append(item_id)

        # Remove all invalid items from the validated dataframe
        if validation_report["removed_items"]:
            # Create mask to keep only valid items
            valid_mask = ~validated_df["item"].isin(validation_report["removed_items"])
            validated_df = validated_df[valid_mask]

            logger.warning(
                f"Final validation: Removed {len(validation_report['removed_items'])} invalid items"
            )
            logger.info(
                f"Final validation: {len(validated_df)} items remain valid out of {len(self.df)}"
            )
        else:
            logger.info("Final validation: All items passed validation")

        # Store validation results for reference
        self.validation_report = validation_report

        # CRITICAL: Attempt to re-place removed items with relaxed constraints
        if validation_report["removed_items"] and input_df is None:
            logger.info(
                f"Attempting to re-place {len(validation_report['removed_items'])} removed items with relaxed constraints"
            )

            # Get the removed items from the original dataframe
            removed_items_df = self.df[
                self.df["item"].isin(validation_report["removed_items"])
            ].copy()

            # Track re-placed items
            re_placed_items = []
            still_unplaced = []

            # Sort removed items by volume (larger items first) for better placement chances
            removed_items_df["volume"] = (
                removed_items_df["width"] * removed_items_df["length"] * removed_items_df["height"]
            )
            removed_items_df = removed_items_df.sort_values("volume", ascending=False)

            for _, item in removed_items_df.iterrows():
                item_id = item["item"]
                logger.info(f"Attempting to re-place item {item_id} with relaxed constraints")

                # Try multiple placement strategies with increasingly relaxed constraints
                position = None

                # Strategy 1: Try optimal horizontal spread placement (NEW - highest priority)
                position = self._find_position_with_optimal_spread(
                    item, validated_df, min_support_threshold=0.2
                )
                if position:
                    logger.info(f"Re-placed item {item_id} with optimal horizontal spread")

                # Strategy 2: Try existing z-levels with relaxed support
                if not position:
                    for z_level in sorted(validated_df["z"].unique()):
                        if z_level + item["height"] <= self.pallet_dims.height:
                            position = self._find_position_at_existing_z_relaxed(
                                item, validated_df, z_level, min_support_threshold=0.2
                            )
                            if position:
                                logger.info(
                                    f"Re-placed item {item_id} at existing z-level {z_level} with relaxed support"
                                )
                                break

                # Strategy 3: Try interlocking placement with relaxed support
                if not position:
                    position = self._find_interlocking_position_relaxed(item, validated_df)
                    if position:
                        logger.info(f"Re-placed item {item_id} with interlocking placement")

                # Strategy 4: Try alternative position with very relaxed constraints
                if not position:
                    position = self._find_alternative_position_relaxed(item, validated_df)
                    if position:
                        logger.info(f"Re-placed item {item_id} with alternative placement")

                # Strategy 5: Try ground level placement if item is small enough
                if not position and item["height"] <= 200:  # Small items can go on ground
                    position = self._find_ground_level_position(item, validated_df)
                    if position:
                        logger.info(f"Re-placed item {item_id} at ground level")

                # If position found, add to validated dataframe
                if position:
                    # Validate the position one more time
                    test_x, test_y, test_z = position["x"], position["y"], position["z"]
                    test_width = position.get("width", item["width"])
                    test_length = position.get("length", item["length"])
                    test_height = item["height"]

                    other_items_list = validated_df.to_dict("records")

                    is_valid, errors = self._validate_position_during_placement(
                        test_x,
                        test_y,
                        test_z,
                        test_width,
                        test_length,
                        test_height,
                        other_items_list,
                        0.2,  # Very relaxed support threshold
                    )

                    if is_valid:
                        # Add the item to validated dataframe
                        new_row = item.copy()
                        new_row["x"] = position["x"]
                        new_row["y"] = position["y"]
                        new_row["z"] = position["z"]

                        if position.get("rotated", False):
                            if "width" in position:
                                new_row["width"] = position["width"]
                            if "length" in position:
                                new_row["length"] = position["length"]

                        validated_df = pd.concat(
                            [validated_df, pd.DataFrame([new_row])], ignore_index=True
                        )
                        re_placed_items.append(item_id)

                        logger.info(
                            f"Successfully re-placed item {item_id} at ({position['x']}, {position['y']}, {position['z']})"
                        )
                    else:
                        logger.warning(
                            f"Re-placement position for item {item_id} failed final validation: {', '.join(errors)}"
                        )
                        still_unplaced.append(item_id)
                else:
                    logger.warning(
                        f"Could not find any valid position for item {item_id} even with relaxed constraints"
                    )
                    still_unplaced.append(item_id)

            # Update validation report
            if re_placed_items:
                logger.info(
                    f"Successfully re-placed {len(re_placed_items)} items: {re_placed_items}"
                )
                # Remove re-placed items from removed_items list
                validation_report["removed_items"] = [
                    item
                    for item in validation_report["removed_items"]
                    if item not in re_placed_items
                ]

            if still_unplaced:
                logger.warning(f"Could not re-place {len(still_unplaced)} items: {still_unplaced}")
                validation_report["unreplaced_items"] = still_unplaced

        # After validation, try to optimize high items if there are valid items
        # Only do this for the main validation call (when input_df is None)
        # and when height optimization is enabled
        if (
            len(validated_df) > 0
            and input_df is None
            and hasattr(self, "optimize_height")
            and self.optimize_height
        ):
            # Only attempt height optimization if no items were removed during validation
            if len(validation_report["removed_items"]) == 0:
                optimized_df = self._optimize_high_items(validated_df)

                # Make sure optimized_df is not the same object as validated_df to prevent reference issues
                if id(optimized_df) != id(validated_df):
                    validated_df = optimized_df

        # Update the main dataframe to the validated version
        # Only do this for the main validation call (when input_df is None)
        if input_df is None:
            self.validated_df = validated_df

        return validated_df, validation_report

    def _find_side_support_maximizing_position(
        self, item, processed_items, current_layer, current_layer_height
    ):
        """Find a position that maximizes side support by prioritizing contact with multiple items."""
        # Extract item dimensions
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records") if not processed_items.empty else []

        # Initialize variables to track best position
        best_position = None
        best_score = float("-inf")  # For side support, higher is better

        # Get unique z-levels to consider
        z_levels = sorted(set(item["z"] for item in items_list)) if items_list else [0]
        if current_layer_height > 0 and current_layer_height not in z_levels:
            z_levels.append(current_layer_height)

        # Use finer grid for better precision in finding optimal side contacts
        grid_step = 5  # Much finer grid than before

        # For each z-level, thoroughly check potential positions
        for z_level in z_levels:
            # Skip if this would exceed pallet height
            if z_level + height > self.pallet_dims.height:
                continue

            # Get all items at this z-level for targeted positioning
            items_at_level = [item for item in items_list if abs(item["z"] - z_level) < 1.0]

            # If no items at this level, check just a few strategic positions
            if not items_at_level:
                # Check origin and corners if z_level is 0 (first layer)
                if abs(z_level) < 0.1:
                    test_positions = [
                        (0, 0),  # Origin
                        (0, max(0, self.pallet_dims.length - length)),  # Bottom-right
                        (max(0, self.pallet_dims.width - width), 0),  # Top-left
                        (
                            max(0, self.pallet_dims.width - width),
                            max(0, self.pallet_dims.length - length),
                        ),  # Top-right
                    ]

                    for x, y in test_positions:
                        if self._is_valid_position(
                            x, y, z_level, width, length, height, items_list
                        ):
                            return {"x": x, "y": y, "z": z_level, "strategic_position": True}
                continue

            # Create targeted search positions near existing items
            search_positions = []

            # Add positions adjacent to existing items (specifically targeting side contacts)
            for existing_item in items_at_level:
                # Try positions that would create side contacts
                adjacent_tests = [
                    # Direct adjacency positions (full side contact)
                    {"x": existing_item["x"] + existing_item["width"], "y": existing_item["y"]},
                    {"x": existing_item["x"] - width, "y": existing_item["y"]},
                    {"x": existing_item["x"], "y": existing_item["y"] + existing_item["length"]},
                    {"x": existing_item["x"], "y": existing_item["y"] - length},
                    # Corner adjacency positions
                    {
                        "x": existing_item["x"] + existing_item["width"],
                        "y": existing_item["y"] + existing_item["length"],
                    },
                    {"x": existing_item["x"] - width, "y": existing_item["y"] - length},
                    {
                        "x": existing_item["x"] + existing_item["width"],
                        "y": existing_item["y"] - length,
                    },
                    {
                        "x": existing_item["x"] - width,
                        "y": existing_item["y"] + existing_item["length"],
                    },
                    # Offset positions that might create partial side contacts
                    {
                        "x": existing_item["x"] + existing_item["width"] - width / 2,
                        "y": existing_item["y"],
                    },
                    {"x": existing_item["x"] - width / 2, "y": existing_item["y"]},
                    {
                        "x": existing_item["x"],
                        "y": existing_item["y"] + existing_item["length"] - length / 2,
                    },
                    {"x": existing_item["x"], "y": existing_item["y"] - length / 2},
                ]

                search_positions.extend(adjacent_tests)

            # For each potential position, calculate side support score
            for pos in search_positions:
                x, y = pos["x"], pos["y"]
                z = z_level

                # Skip invalid positions
                if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                    continue

                if self._check_overlap(x, y, z, width, length, height, items_list):
                    continue

                # Check for bottom support
                support_percentage, is_supported = self._calculate_support(
                    x,
                    y,
                    z,
                    width,
                    length,
                    items_list,
                    0.3,  # Relaxed threshold for side support focus
                )

                if not is_supported and abs(z) > 0.1:  # If not on ground and not supported
                    continue

                # Calculate comprehensive side support score
                side_contact_data = self._calculate_enhanced_side_contact(
                    x, y, z, width, length, height, items_list
                )

                # Calculate horizontal spread score
                spread_data = self._calculate_horizontal_spread_score(
                    x, y, z, width, length, height, items_list
                )

                # Extract metrics from side contact data
                contact_area_ratio = side_contact_data["contact_area_ratio"]
                num_contact_items = side_contact_data["num_contact_items"]
                num_contact_faces = side_contact_data["num_contact_faces"]
                max_face_coverage = side_contact_data["max_face_coverage"]

                # Calculate position score with balanced side support and horizontal spread
                side_support_score = (
                    contact_area_ratio * 200  # Base contact area
                    + num_contact_items * 300  # Bonus for each item in contact (interlocking)
                    + num_contact_faces * 150  # Bonus for contacting multiple faces
                    + max_face_coverage * 250  # Bonus for having one well-supported face
                )

                # Combine side support and horizontal spread
                combined_score = (
                    side_support_score * 0.6  # 60% weight to side support
                    + spread_data["spread_score"] * 0.4  # 40% weight to horizontal spread
                    - (z * 30)  # Mild height penalty
                    - (x + y) / 20  # Very mild origin preference
                )

                # If this position has better combined score, update best position
                if combined_score > best_score:
                    best_score = combined_score
                    best_position = {
                        "x": x,
                        "y": y,
                        "z": z,
                        "score": combined_score,
                        "support": support_percentage,
                        "side_contact": contact_area_ratio,
                        "contact_items": num_contact_items,
                        "contact_faces": num_contact_faces,
                        "spread_score": spread_data["spread_score"],
                        "spread_data": spread_data,
                    }

            # If we found a good position at this level, return it directly
            if best_position and best_score > 1000:  # Threshold for "good enough" side support
                return best_position

            # If no targeted positions worked well, try a grid search with side support focus
            if not best_position or best_score < 500:
                for x in range(0, int(self.pallet_dims.width - width + 1), grid_step):
                    for y in range(0, int(self.pallet_dims.length - length + 1), grid_step):
                        # Skip invalid positions
                        if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                            continue

                        if self._check_overlap(x, y, z, width, length, height, items_list):
                            continue

                        # Check for bottom support
                        support_percentage, is_supported = self._calculate_support(
                            x, y, z, width, length, items_list, 0.3
                        )

                        if not is_supported and abs(z) > 0.1:
                            continue

                        # Calculate comprehensive side support
                        side_contact_data = self._calculate_enhanced_side_contact(
                            x, y, z, width, length, height, items_list
                        )

                        contact_area_ratio = side_contact_data["contact_area_ratio"]
                        num_contact_items = side_contact_data["num_contact_items"]
                        num_contact_faces = side_contact_data["num_contact_faces"]
                        max_face_coverage = side_contact_data["max_face_coverage"]

                        # Calculate position score for grid search with balanced approach
                        side_support_score = (
                            contact_area_ratio * 200
                            + num_contact_items * 300
                            + num_contact_faces * 150
                            + max_face_coverage * 250
                        )

                        # Combine side support and horizontal spread for grid positions
                        grid_score = (
                            side_support_score * 0.5  # 50% weight to side support
                            + spread_data["spread_score"]
                            * 0.5  # 50% weight to horizontal spread (higher for grid)
                            - (z * 30)
                            - (x + y) / 20
                        )

                        if grid_score > best_score:
                            best_score = grid_score
                            best_position = {
                                "x": x,
                                "y": y,
                                "z": z,
                                "score": grid_score,
                                "support": support_percentage,
                                "side_contact": contact_area_ratio,
                                "contact_items": num_contact_items,
                                "contact_faces": num_contact_faces,
                                "spread_score": spread_data["spread_score"],
                                "spread_data": spread_data,
                                "grid_position": True,
                            }

        return best_position

    def _calculate_enhanced_side_contact(self, x, y, z, width, length, height, items_list):
        """Calculate detailed side contact metrics for comprehensive evaluation."""
        # Define the 6 faces of the box
        faces = {
            "left": {"normal": (-1, 0, 0), "area": length * height},
            "right": {"normal": (1, 0, 0), "area": length * height},
            "front": {"normal": (0, -1, 0), "area": width * height},
            "back": {"normal": (0, 1, 0), "area": width * height},
            "bottom": {"normal": (0, 0, -1), "area": width * length},
            "top": {"normal": (0, 0, 1), "area": width * length},
        }

        # Track contact metrics
        contact_items = set()
        face_contacts = {face: 0 for face in faces}
        face_items = {face: set() for face in faces}
        total_side_area = 2 * (width * height + length * height)  # Exclude top/bottom
        total_contact_area = 0

        # Check contact for each face against each item
        for other_item in items_list:
            # Left face
            if abs(other_item["x"] + other_item["width"] - x) < 1.0:  # More forgiving threshold
                y_overlap = max(
                    0,
                    min(y + length, other_item["y"] + other_item["length"])
                    - max(y, other_item["y"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                contact_area = y_overlap * z_overlap

                if contact_area > 0:
                    face_contacts["left"] += contact_area
                    face_items["left"].add(other_item["item"])
                    contact_items.add(other_item["item"])
                    total_contact_area += contact_area

            # Right face
            if abs(other_item["x"] - (x + width)) < 1.0:
                y_overlap = max(
                    0,
                    min(y + length, other_item["y"] + other_item["length"])
                    - max(y, other_item["y"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                contact_area = y_overlap * z_overlap

                if contact_area > 0:
                    face_contacts["right"] += contact_area
                    face_items["right"].add(other_item["item"])
                    contact_items.add(other_item["item"])
                    total_contact_area += contact_area

            # Front face
            if abs(other_item["y"] + other_item["length"] - y) < 1.0:
                x_overlap = max(
                    0,
                    min(x + width, other_item["x"] + other_item["width"]) - max(x, other_item["x"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                contact_area = x_overlap * z_overlap

                if contact_area > 0:
                    face_contacts["front"] += contact_area
                    face_items["front"].add(other_item["item"])
                    contact_items.add(other_item["item"])
                    total_contact_area += contact_area

            # Back face
            if abs(other_item["y"] - (y + length)) < 1.0:
                x_overlap = max(
                    0,
                    min(x + width, other_item["x"] + other_item["width"]) - max(x, other_item["x"]),
                )
                z_overlap = max(
                    0,
                    min(z + height, other_item["z"] + other_item["height"])
                    - max(z, other_item["z"]),
                )
                contact_area = x_overlap * z_overlap

                if contact_area > 0:
                    face_contacts["back"] += contact_area
                    face_items["back"].add(other_item["item"])
                    contact_items.add(other_item["item"])
                    total_contact_area += contact_area

        # Calculate advanced metrics
        num_contact_faces = sum(
            1 for face in ["left", "right", "front", "back"] if face_contacts[face] > 0
        )

        # Find max face coverage - important for stability
        max_face_coverage = 0
        for face in ["left", "right", "front", "back"]:
            face_coverage = (
                face_contacts[face] / faces[face]["area"] if faces[face]["area"] > 0 else 0
            )
            max_face_coverage = max(max_face_coverage, face_coverage)

        # Calculate contact area ratio for side faces only
        contact_area_ratio = total_contact_area / total_side_area if total_side_area > 0 else 0

        # Return comprehensive data
        return {
            "contact_area_ratio": contact_area_ratio,
            "num_contact_items": len(contact_items),
            "num_contact_faces": num_contact_faces,
            "max_face_coverage": max_face_coverage,
            "face_contacts": face_contacts,
            "face_items": face_items,
        }

    # Add this helper method to your CompactBin class
    def _is_valid_position_for_compaction(
        self, x, y, z, width, length, height, items_list, min_support=0.5
    ):
        """More comprehensive check for valid position during compaction,."""
        # First check bounds and overlap
        if not self._is_within_pallet_bounds(x, y, z, width, length, height):
            return False

        if self._check_overlap(x, y, z, width, length, height, items_list):
            return False

        # Skip support check for ground level
        if abs(z) < 0.1:
            return True

        # Check for support
        support_percentage, is_supported = self._calculate_support(
            x, y, z, width, length, items_list, min_support
        )

        return is_supported

    def plot(self, df=None, title="Final Bin Packing", use_validated=True):
        """Return a bin plot without the layers representation."""

        if df is None:
            df = self.validated_df if use_validated and hasattr(self, "validated_df") else self.df

        # Plot the bin
        ax = visualization.get_pallet_plot(self.pallet_dims)
        for _, item in df.iterrows():
            ax = visualization.plot_product(
                ax,
                item["item"],
                utils.Coordinate(item.x, item.y, item.z),
                utils.Dimension(item.width, item.length, item.height),
                self.pallet_dims,
            )
        ax.set_title(title, fontsize=10, y=1.05)
        return ax

    def _try_orientations(self, item, try_fn):
        """Try both orientations (width x length and length x width) for an item."""
        results = []
        for width, length, rotated in [
            (item["width"], item["length"], False),
            (
                (item["length"], item["width"], True)
                if item["width"] != item["length"]
                else (None, None, None)
            ),
        ]:
            if width is not None:
                result = try_fn(width, length, rotated)
                if result:
                    results.append(result)
        if not results:
            return None
        # If results have a 'score', pick the best one; otherwise, just return the first
        if "score" in results[0]:
            return min(results, key=lambda r: r["score"])
        return results[0]

    def _validate_position_during_placement(
        self, x, y, z, width, length, height, other_items_list, min_support_threshold=0.3
    ):
        """Comprehensive validation of a position during placement."""
        error_messages = []

        # Check bounds
        if not self._is_within_pallet_bounds(x, y, z, width, length, height):
            error_messages.append("outside pallet bounds")

        # Check overlaps
        if self._check_overlap(x, y, z, width, length, height, other_items_list):
            error_messages.append("overlaps with existing items")

        # Check support (only if not on ground level)
        if z > 0.1:
            support_percentage, is_supported = self._calculate_support(
                x, y, z, width, length, other_items_list, min_support_threshold
            )
            if not is_supported:
                error_messages.append(f"insufficient support ({support_percentage*100:.1f}%)")

        is_valid = len(error_messages) == 0
        return is_valid, error_messages

    def _find_position_at_existing_z_relaxed(
        self, item, processed_items, z_level, min_support_threshold=0.2
    ):
        """Find a position at existing z-level with relaxed constraints for re-placement."""
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        # Try positions at this z-level with relaxed constraints
        step_size = 25  # Finer grid for re-placement

        for x in range(0, int(self.pallet_dims.width - width + 1), step_size):
            for y in range(0, int(self.pallet_dims.length - length + 1), step_size):
                if not self._is_within_pallet_bounds(x, y, z_level, width, length, height):
                    continue

                if self._check_overlap(x, y, z_level, width, length, height, items_list):
                    continue

                # Check support with relaxed threshold
                support_percentage, is_supported = self._calculate_support(
                    x, y, z_level, width, length, items_list, min_support_threshold
                )

                if is_supported:
                    return {"x": x, "y": y, "z": z_level, "support": support_percentage}

        return None

    def _find_interlocking_position_relaxed(self, item, processed_items):
        """Find interlocking position with relaxed constraints for re-placement."""
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        # Try with relaxed support threshold
        min_support_threshold = 0.2

        # Get all unique z-levels
        z_levels = sorted(set(item["z"] for item in items_list)) if items_list else [0]

        for z_level in z_levels:
            if z_level + height > self.pallet_dims.height:
                continue

            items_at_level = [item for item in items_list if abs(item["z"] - z_level) < 1.0]

            if len(items_at_level) < 2:
                continue

            # Try positions near existing items
            for existing_item in items_at_level:
                adjacent_positions = [
                    {"x": existing_item["x"] + existing_item["width"], "y": existing_item["y"]},
                    {"x": existing_item["x"] - width, "y": existing_item["y"]},
                    {"x": existing_item["x"], "y": existing_item["y"] + existing_item["length"]},
                    {"x": existing_item["x"], "y": existing_item["y"] - length},
                ]

                for pos in adjacent_positions:
                    x, y = pos["x"], pos["y"]
                    z = z_level

                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, min_support_threshold
                    )

                    if is_supported:
                        return {"x": x, "y": y, "z": z, "support": support_percentage}

        return None

    def _find_alternative_position_relaxed(self, item, processed_items):
        """Find alternative position with very relaxed constraints for re-placement."""
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        # Very relaxed support threshold
        min_support_threshold = 0.1

        # Try positions at different z-levels
        z_step = 100  # Larger steps for re-placement
        step_size = 50

        for z in range(0, int(self.pallet_dims.height - height + 1), z_step):
            for x in range(0, int(self.pallet_dims.width - width + 1), step_size):
                for y in range(0, int(self.pallet_dims.length - length + 1), step_size):
                    if not self._is_within_pallet_bounds(x, y, z, width, length, height):
                        continue

                    if self._check_overlap(x, y, z, width, length, height, items_list):
                        continue

                    support_percentage, is_supported = self._calculate_support(
                        x, y, z, width, length, items_list, min_support_threshold
                    )

                    if is_supported:
                        return {"x": x, "y": y, "z": z, "support": support_percentage}

        return None

    def _find_ground_level_position(self, item, processed_items):
        """Find a position at ground level (z=0) for small items."""
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        z_level = 0
        step_size = 25

        for x in range(0, int(self.pallet_dims.width - width + 1), step_size):
            for y in range(0, int(self.pallet_dims.length - length + 1), step_size):
                if not self._is_within_pallet_bounds(x, y, z_level, width, length, height):
                    continue

                if self._check_overlap(x, y, z_level, width, length, height, items_list):
                    continue

                # Ground level items have 100% support
                return {"x": x, "y": y, "z": z_level, "support": 1.0}

        return None

    def _calculate_horizontal_spread_score(self, x, y, z, width, length, height, items_list):
        """Calculate a comprehensive horizontal spread score that measures how well."""
        # Get all items at the same z-level (or very close)
        items_at_level = [item for item in items_list if abs(item["z"] - z) < 1.0]

        # Calculate current coverage area at this z-level
        if not items_at_level:
            # First item at this level - place it strategically
            return self._calculate_first_item_spread_score(x, y, width, length)

        # Calculate current bounding box of items at this level
        min_x = min(item["x"] for item in items_at_level)
        max_x = max(item["x"] + item["width"] for item in items_at_level)
        min_y = min(item["y"] for item in items_at_level)
        max_y = max(item["y"] + item["length"] for item in items_at_level)

        current_width = max_x - min_x
        current_length = max_y - min_y
        current_area = current_width * current_length

        # Calculate new bounding box if this item is added
        new_min_x = min(min_x, x)
        new_max_x = max(max_x, x + width)
        new_min_y = min(min_y, y)
        new_max_y = max(max_y, y + length)

        new_width = new_max_x - new_min_x
        new_length = new_max_y - new_min_y
        new_area = new_width * new_length

        # Calculate area expansion
        area_expansion = new_area - current_area
        area_expansion_ratio = area_expansion / current_area if current_area > 0 else 1.0

        # Calculate coverage improvement
        pallet_area = self.pallet_dims.width * self.pallet_dims.length
        current_coverage = current_area / pallet_area if pallet_area > 0 else 0
        new_coverage = new_area / pallet_area if pallet_area > 0 else 0
        coverage_improvement = new_coverage - current_coverage

        # Calculate distance from existing items (encourage spreading)
        min_distance_to_existing = float("inf")
        total_distance = 0
        num_items = len(items_at_level)

        for item in items_at_level:
            # Calculate center-to-center distance
            item_center_x = item["x"] + item["width"] / 2
            item_center_y = item["y"] + item["length"] / 2
            new_center_x = x + width / 2
            new_center_y = y + length / 2

            distance = np.sqrt(
                (new_center_x - item_center_x) ** 2 + (new_center_y - item_center_y) ** 2
            )
            min_distance_to_existing = min(min_distance_to_existing, distance)
            total_distance += distance

        avg_distance = total_distance / num_items if num_items > 0 else 0

        # Calculate quadrant distribution
        pallet_center_x = self.pallet_dims.width / 2
        pallet_center_y = self.pallet_dims.length / 2

        # Count items in each quadrant
        quadrants = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        for item in items_at_level:
            item_center_x = item["x"] + item["width"] / 2
            item_center_y = item["y"] + item["length"] / 2

            if item_center_x < pallet_center_x:
                if item_center_y < pallet_center_y:
                    quadrants["bottom_left"] += 1
                else:
                    quadrants["top_left"] += 1
            else:
                if item_center_y < pallet_center_y:
                    quadrants["bottom_right"] += 1
                else:
                    quadrants["top_right"] += 1

        # Determine which quadrant the new item would be in
        new_center_x = x + width / 2
        new_center_y = y + length / 2

        if new_center_x < pallet_center_x:
            if new_center_y < pallet_center_y:
                target_quadrant = "bottom_left"
            else:
                target_quadrant = "top_left"
        else:
            if new_center_y < pallet_center_y:
                target_quadrant = "bottom_right"
            else:
                target_quadrant = "top_right"

        # Calculate quadrant balance score
        current_quadrant_count = quadrants[target_quadrant]
        quadrant_balance_score = 1.0 / (
            current_quadrant_count + 1
        )  # Prefer less populated quadrants

        # Calculate edge utilization
        edge_distance = min(
            x,
            y,  # Distance from left and bottom edges
            self.pallet_dims.width - (x + width),  # Distance from right edge
            self.pallet_dims.length - (y + length),  # Distance from top edge
        )
        edge_utilization_score = 1.0 / (edge_distance + 1)  # Prefer positions closer to edges

        # Calculate overall spread score
        spread_score = (
            area_expansion_ratio * 300  # Area expansion (most important)
            + coverage_improvement * 500  # Coverage improvement
            + min_distance_to_existing * 2  # Distance from existing items
            + avg_distance * 1.5  # Average distance
            + quadrant_balance_score * 200  # Quadrant balance
            + edge_utilization_score * 100  # Edge utilization
        )

        return {
            "spread_score": spread_score,
            "area_expansion_ratio": area_expansion_ratio,
            "coverage_improvement": coverage_improvement,
            "min_distance_to_existing": min_distance_to_existing,
            "avg_distance": avg_distance,
            "quadrant_balance_score": quadrant_balance_score,
            "edge_utilization_score": edge_utilization_score,
            "current_coverage": current_coverage,
            "new_coverage": new_coverage,
            "target_quadrant": target_quadrant,
            "quadrant_population": current_quadrant_count,
        }

    def _calculate_first_item_spread_score(self, x, y, width, length):
        """Calculate spread score for the first item at a z-level."""
        # Calculate distance from center
        pallet_center_x = self.pallet_dims.width / 2
        pallet_center_y = self.pallet_dims.length / 2
        item_center_x = x + width / 2
        item_center_y = y + length / 2

        center_distance = np.sqrt(
            (item_center_x - pallet_center_x) ** 2 + (item_center_y - pallet_center_y) ** 2
        )

        # Calculate distance from edges
        edge_distance = min(
            x,
            y,  # Distance from left and bottom edges
            self.pallet_dims.width - (x + width),  # Distance from right edge
            self.pallet_dims.length - (y + length),  # Distance from top edge
        )

        # Calculate area coverage
        item_area = width * length
        pallet_area = self.pallet_dims.width * self.pallet_dims.length
        coverage = item_area / pallet_area if pallet_area > 0 else 0

        # Prefer positions that:
        # 1. Are not at the very center (leave room for other items)
        # 2. Are not too close to edges (allow for expansion)
        # 3. Have good coverage
        center_penalty = max(0, 100 - center_distance)  # Penalty for being too close to center
        edge_bonus = max(0, 50 - edge_distance)  # Bonus for being near edges but not too close
        coverage_bonus = coverage * 200

        spread_score = coverage_bonus + edge_bonus - center_penalty

        return {
            "spread_score": spread_score,
            "area_expansion_ratio": 1.0,  # First item creates all the area
            "coverage_improvement": coverage,
            "min_distance_to_existing": float("inf"),
            "avg_distance": 0,
            "quadrant_balance_score": 1.0,
            "edge_utilization_score": 1.0 / (edge_distance + 1),
            "current_coverage": 0,
            "new_coverage": coverage,
            "target_quadrant": "first_item",
            "quadrant_population": 0,
        }

    def _find_position_with_optimal_spread(self, item, processed_items, min_support_threshold=0.2):
        """Find a position that maximizes horizontal spread for re-placement of removed items."""
        width, length, height = item["width"], item["length"], item["height"]
        items_list = processed_items.to_dict("records")

        # Get all unique z-levels
        z_levels = sorted(set(item["z"] for item in items_list)) if items_list else [0]

        best_position = None
        best_spread_score = float("-inf")

        # Try each z-level
        for z_level in z_levels:
            if z_level + height > self.pallet_dims.height:
                continue

            # Try strategic positions that maximize spread
            strategic_positions = [
                # Corner positions
                (0, 0),
                (0, max(0, self.pallet_dims.length - length)),
                (max(0, self.pallet_dims.width - width), 0),
                (max(0, self.pallet_dims.width - width), max(0, self.pallet_dims.length - length)),
                # Edge positions
                (0, self.pallet_dims.length // 2 - length // 2),
                (self.pallet_dims.width // 2 - width // 2, 0),
                (self.pallet_dims.width - width, self.pallet_dims.length // 2 - length // 2),
                (self.pallet_dims.width // 2 - width // 2, self.pallet_dims.length - length),
                # Center positions (if no items there)
                (
                    self.pallet_dims.width // 2 - width // 2,
                    self.pallet_dims.length // 2 - length // 2,
                ),
            ]

            for x, y in strategic_positions:
                if not self._is_within_pallet_bounds(x, y, z_level, width, length, height):
                    continue

                if self._check_overlap(x, y, z_level, width, length, height, items_list):
                    continue

                # Check support
                support_percentage, is_supported = self._calculate_support(
                    x, y, z_level, width, length, items_list, min_support_threshold
                )

                if is_supported:
                    # Calculate horizontal spread score
                    spread_data = self._calculate_horizontal_spread_score(
                        x, y, z_level, width, length, height, items_list
                    )

                    if spread_data["spread_score"] > best_spread_score:
                        best_spread_score = spread_data["spread_score"]
                        best_position = {
                            "x": x,
                            "y": y,
                            "z": z_level,
                            "support": support_percentage,
                            "spread_score": spread_data["spread_score"],
                            "spread_data": spread_data,
                        }

        # If no strategic positions work, try grid search with spread focus
        if not best_position:
            grid_step = 25
            for z_level in z_levels:
                if z_level + height > self.pallet_dims.height:
                    continue

                for x in range(0, int(self.pallet_dims.width - width + 1), grid_step):
                    for y in range(0, int(self.pallet_dims.length - length + 1), grid_step):
                        if not self._is_within_pallet_bounds(x, y, z_level, width, length, height):
                            continue

                        if self._check_overlap(x, y, z_level, width, length, height, items_list):
                            continue

                        support_percentage, is_supported = self._calculate_support(
                            x, y, z_level, width, length, items_list, min_support_threshold
                        )

                        if is_supported:
                            spread_data = self._calculate_horizontal_spread_score(
                                x, y, z_level, width, length, height, items_list
                            )

                            if spread_data["spread_score"] > best_spread_score:
                                best_spread_score = spread_data["spread_score"]
                                best_position = {
                                    "x": x,
                                    "y": y,
                                    "z": z_level,
                                    "support": support_percentage,
                                    "spread_score": spread_data["spread_score"],
                                    "spread_data": spread_data,
                                }

        return best_position


class CompactBinPool:
    """A collection of compact bins with advanced validation and analysis capabilities."""

    def __init__(self, bin_pool, use_sequential=True, validate_final=True):
        """Initialize a pool of compact bins with validation support."""
        self.compact_bins = []
        self._original_bin_pool = bin_pool
        self.validation_summary = None

        # Create compact bins with validation
        for bin_idx, bin in enumerate(bin_pool):
            logger.info(f"Creating compact bin {bin_idx+1}/{len(bin_pool)}")
            self.compact_bins.append(
                CompactBin(
                    bin.to_dataframe(),
                    bin_pool.pallet_dims,
                    use_sequential=use_sequential,
                    validate_final=validate_final,
                )
            )

        # Collect validation reports from all bins
        if validate_final:
            self._compile_validation_summary()

    def _compile_validation_summary(self):
        """Compile validation results from all bins into a summary report."""
        summary = {
            "total_items": 0,
            "valid_items": 0,
            "removed_items": 0,
            "bins_with_issues": 0,
            "overlapping_items_count": 0,
            "insufficient_support_count": 0,
            "bin_reports": [],
        }

        for bin_idx, bin in enumerate(self.compact_bins):
            if not hasattr(bin, "validation_report"):
                continue

            # Count items
            bin_item_count = len(bin.df)
            valid_item_count = (
                len(bin.validated_df) if bin.validated_df is not None else bin_item_count
            )
            removed_count = bin_item_count - valid_item_count

            summary["total_items"] += bin_item_count
            summary["valid_items"] += valid_item_count

            # Record bin-specific issues
            if removed_count > 0:
                summary["bins_with_issues"] += 1
                summary["removed_items"] += removed_count
                summary["overlapping_items_count"] += len(
                    bin.validation_report["overlapping_items"]
                )
                summary["insufficient_support_count"] += len(
                    bin.validation_report["insufficient_support_items"]
                )

                # Add detailed bin report
                summary["bin_reports"].append(
                    {
                        "bin_idx": bin_idx,
                        "total_items": bin_item_count,
                        "valid_items": valid_item_count,
                        "removed_items": removed_count,
                        "overlapping_items": bin.validation_report["overlapping_items"],
                        "insufficient_support_items": bin.validation_report[
                            "insufficient_support_items"
                        ],
                    }
                )

        # Calculate overall metrics
        if summary["total_items"] > 0:
            summary["valid_percentage"] = (summary["valid_items"] / summary["total_items"]) * 100
        else:
            summary["valid_percentage"] = 0

        self.validation_summary = summary

        # Log validation summary
        logger.info(
            f"Validation summary: {summary['valid_items']}/{summary['total_items']} items valid ({summary['valid_percentage']:.2f}%)"
        )
        if summary["removed_items"] > 0:
            logger.warning(
                f"Removed {summary['removed_items']} items: {summary['overlapping_items_count']} overlapping, {summary['insufficient_support_count']} insufficient support"
            )
            logger.warning(
                f"Issues found in {summary['bins_with_issues']}/{len(self.compact_bins)} bins"
            )

    def get_original_bin_pool(self):
        """Get the original uncompacted bin pool."""
        return self._original_bin_pool

    def get_original_layer_pool(self):
        """Get the layer pool used to build bins prior to compacting."""
        return self._original_bin_pool.layer_pool

    def plot(self, use_validated=True):
        """Generate 3D visualizations for all compact bins in the pool."""
        axs = []
        for bin_idx, bin in enumerate(self.compact_bins):
            title = f"Bin {bin_idx+1} - {'Validated' if use_validated else 'Original'} Solution"
            ax = bin.plot(title=title, use_validated=use_validated)
            axs.append(ax)
        return axs

    def to_dataframe(self, use_validated=True):
        """Convert all compact bins to a single pandas DataFrame."""
        dfs = []
        for i, compact_bin in enumerate(self.compact_bins):
            # Use either validated or original dataframe
            if (
                use_validated
                and hasattr(compact_bin, "validated_df")
                and compact_bin.validated_df is not None
            ):
                df = compact_bin.validated_df.copy()
            else:
                df = compact_bin.df.copy()

            df["bin"] = i
            dfs.append(df)

        if dfs:
            combined = pd.concat(dfs, axis=0)
            duplicated_items = combined[combined.duplicated(subset=["item"], keep=False)][
                "item"
            ].unique()
            if len(duplicated_items) > 0:
                logger.warning(
                    f"Duplicate packed item ids detected in CompactBinPool output: {list(duplicated_items)}"
                )
            return combined
        else:
            # Return empty dataframe with expected columns if no bins
            return pd.DataFrame(columns=["item", "x", "y", "z", "width", "length", "height", "bin"])

    def get_bin_utilization(self, use_validated=True):
        """Calculate comprehensive utilization metrics for all bins."""
        metrics = {
            "bins": [],
            "overall": {"total_items": 0, "total_volume": 0, "total_capacity": 0},
        }

        pallet_volume = (
            self._original_bin_pool.pallet_dims.width
            * self._original_bin_pool.pallet_dims.length
            * self._original_bin_pool.pallet_dims.height
        )

        for bin_idx, bin in enumerate(self.compact_bins):
            # Get appropriate dataframe
            if use_validated and hasattr(bin, "validated_df") and bin.validated_df is not None:
                df = bin.validated_df
            else:
                df = bin.df

            # Calculate metrics
            items_count = len(df)
            item_volume = sum(row.width * row.length * row.height for _, row in df.iterrows())
            volume_utilization = (item_volume / pallet_volume) * 100 if pallet_volume > 0 else 0

            # Calculate max height
            max_height = 0
            if not df.empty:
                max_height = max(row.z + row.height for _, row in df.iterrows())

            # Store bin metrics
            metrics["bins"].append(
                {
                    "bin_idx": bin_idx,
                    "items_count": items_count,
                    "volume_utilization": volume_utilization,
                    "max_height": max_height,
                }
            )

            # Update overall metrics
            metrics["overall"]["total_items"] += items_count
            metrics["overall"]["total_volume"] += item_volume
            metrics["overall"]["total_capacity"] += pallet_volume

        # Calculate overall utilization
        if metrics["overall"]["total_capacity"] > 0:
            metrics["overall"]["volume_utilization"] = (
                metrics["overall"]["total_volume"] / metrics["overall"]["total_capacity"]
            ) * 100
        else:
            metrics["overall"]["volume_utilization"] = 0

        return metrics
