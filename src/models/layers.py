import copy

import numpy as np
import pandas as pd
from loguru import logger

from src.models import maxrects, superitems
from src.utils import utils, visualization


class Layer:
    """Horizontal layer of superitems with similar heights."""

    def __init__(self, superitems_pool, superitems_coords, pallet_dims):
        """Initialize a layer with superitems and their coordinates."""
        self.superitems_pool = superitems_pool
        self.superitems_coords = superitems_coords
        self.pallet_dims = pallet_dims

    @property
    def height(self):
        """Get the height of the layer (maximum superitem height)."""
        return self.superitems_pool.get_max_height()

    @property
    def volume(self):
        """Calculate the total volume of all superitems in the layer."""
        return sum(s.volume for s in self.superitems_pool)

    @property
    def area(self):
        """Calculate the total base area of all superitems in the layer."""
        return sum(s.width * s.length for s in self.superitems_pool)

    def is_empty(self):
        """Check if the layer contains any superitems."""
        return len(self.superitems_pool) == 0 and len(self.superitems_coords) == 0

    def subset(self, superitem_indices):
        """Create a new layer containing only specified superitems."""
        new_spool = self.superitems_pool.subset(superitem_indices)
        new_scoords = [c for i, c in enumerate(self.superitems_coords) if i in superitem_indices]
        return Layer(new_spool, new_scoords, self.pallet_dims)

    def difference(self, superitem_indices):
        """Create a new layer excluding specified superitems."""
        new_spool = self.superitems_pool.difference(superitem_indices)
        new_scoords = [
            c for i, c in enumerate(self.superitems_coords) if i not in superitem_indices
        ]
        return Layer(new_spool, new_scoords, self.pallet_dims)

    def get_items_coords(self):
        """Get 3D coordinates for all individual items in the layer."""
        items_coords = {}
        seen_items = set()
        for s, c in zip(self.superitems_pool, self.superitems_coords):
            coords = s.get_items_coords(width=c.x, length=c.y, height=c.z)  # Use coord.z
            for item_id, coord in coords.items():
                if item_id not in seen_items:
                    items_coords[item_id] = coord
                    seen_items.add(item_id)
        return items_coords

    def assign_coordinates(self, base_z=0, sort_by_density=True):
        """Assign 2D coordinates to superitems using bottom-left placement strategy."""
        # If coordinates are already assigned, just update z
        if len(self.superitems_coords) == len(self.superitems_pool):
            for coord in self.superitems_coords:
                coord.z = base_z
            logger.debug(
                f"Updated z-coordinates to base_z={base_z} for {len(self.superitems_coords)} superitems"
            )
            return

        # Reset coordinates
        self.superitems_coords = []

        # Create 2D grid to track occupied spaces
        grid_resolution = 1  # Use 1 unit resolution for better performance
        grid_width = int(self.pallet_dims.width * grid_resolution)
        grid_length = int(self.pallet_dims.length * grid_resolution)
        occupied = np.zeros((grid_width, grid_length), dtype=bool)

        # Sort by density or area for better packing
        if sort_by_density:
            # Calculate density for each superitem (volume / area)
            densities = []
            for s in self.superitems_pool:
                volume = s.volume
                area = s.width * s.length
                density = volume / area if area > 0 else 0
                densities.append(density)

            # Sort by density (highest first) for better packing efficiency
            sorted_indices = utils.argsort(densities, reverse=True)
            sorted_superitems = [self.superitems_pool[i] for i in sorted_indices]

            logger.debug(
                f"Sorted layer items by density (range: {min(densities):.2f}-{max(densities):.2f})"
            )
        else:
            # Original area-based sorting
            sorted_indices = utils.argsort(
                [s.width * s.length for s in self.superitems_pool], reverse=True
            )
            sorted_superitems = [self.superitems_pool[i] for i in sorted_indices]

        # Place items using optimized bottom-left strategy
        placed_indices = []

        for idx, superitem in enumerate(sorted_superitems):
            w, l, h = superitem.width, superitem.length, superitem.height

            # Skip if height exceeds remaining pallet height
            if base_z + h > self.pallet_dims.height:
                logger.debug(
                    f"Superitem {superitem.id} height {h} exceeds pallet height at base_z={base_z}"
                )
                continue

            # Convert superitem dimensions to grid units
            grid_w = int(w * grid_resolution)
            grid_l = int(l * grid_resolution)

            # Find the bottom-left position with good adjacency
            placed = False
            best_x, best_y = 0, 0
            best_score = float("-inf")

            # Use a stride to skip some positions for efficiency
            stride = max(1, min(grid_w, grid_l) // 4)  # Stride based on item size

            for start_y in range(0, grid_length - grid_l + 1, stride):
                for start_x in range(0, grid_width - grid_w + 1, stride):
                    if not self._grid_overlaps(start_x, start_y, grid_w, grid_l, occupied):
                        # Use a simplified scoring system

                        # Distance from origin (bottom-left priority)
                        distance = (start_x**2 + start_y**2) ** 0.5
                        distance_score = 1.0 - (distance / (grid_width + grid_length))

                        # Wall contact (simplified)
                        wall_contact = 0
                        if start_x == 0:
                            wall_contact += 1
                        if start_y == 0:
                            wall_contact += 1
                        wall_score = wall_contact * 0.2

                        # Calculate total score
                        score = distance_score + wall_score

                        if score > best_score:
                            best_score = score
                            best_x, best_y = start_x, start_y
                            placed = True

            if placed:
                # Convert grid coordinates back to actual coordinates
                actual_x = best_x / grid_resolution
                actual_y = best_y / grid_resolution

                # Create coordinate
                coord = utils.Coordinate(actual_x, actual_y, base_z)
                self.superitems_coords.append(coord)
                placed_indices.append(sorted_indices[idx])

                # Mark area as occupied
                self._grid_mark_occupied(best_x, best_y, grid_w, grid_l, occupied)
            else:
                logger.warning(
                    f"Could not place superitem {superitem.id} (w={w}, l={l}, h={h}) at base_z={base_z}"
                )

        # Recreate superitems_pool to match with placed items only
        if placed_indices:
            self.superitems_pool = self.superitems_pool.subset(placed_indices)

            logger.debug(
                f"Placed {len(self.superitems_coords)}/{len(sorted_superitems)} superitems"
            )
        else:
            logger.warning(f"Could not place any superitems at base_z={base_z}")

            # Make a final attempt with simplified placement (just at origin)
            if (
                sorted_superitems
                and sorted_superitems[0].height + base_z <= self.pallet_dims.height
            ):
                self.superitems_coords.append(utils.Coordinate(0, 0, base_z))
                self.superitems_pool = self.superitems_pool.subset([sorted_indices[0]])
                logger.debug(f"Placed one superitem as fallback")

    def _grid_overlaps(self, x, y, width, length, occupied):
        """Check if a rectangular area in the grid is occupied."""
        if x + width > occupied.shape[0] or y + length > occupied.shape[1]:
            return True
        return np.any(occupied[x : x + width, y : y + length])

    def _grid_mark_occupied(self, x, y, width, length, occupied):
        """Mark a rectangular area in the grid as occupied."""
        if x + width <= occupied.shape[0] and y + length <= occupied.shape[1]:
            occupied[x : x + width, y : y + length] = True

    def _overlaps(self, x, y, width, length, occupied):
        x_end = min(x + width, self.pallet_dims.width)
        y_end = min(y + length, self.pallet_dims.length)
        return np.any(occupied[x:x_end, y:y_end])

    def _mark_occupied(self, x, y, width, length, occupied):
        x_end = min(x + width, self.pallet_dims.width)
        y_end = min(y + length, self.pallet_dims.length)
        occupied[x:x_end, y:y_end] = True

    def _check_overlap(self, coord1, coord2, dim1, dim2):
        """Check if two coordinates with dimensions overlap."""
        x1, y1, z1 = coord1.x, coord1.y, coord1.z
        x2, y2, z2 = coord2.x, coord2.y, coord2.z
        return (
            x1 < x2 + dim2.width
            and x2 < x1 + dim1.width
            and y1 < y2 + dim2.length
            and y2 < y1 + dim1.length
            and z1 < z2 + dim1.height
            and z2 < z1 + dim1.height
        )

    def get_items_dims(self):
        items_dims = {}
        seen_items = set()  # Track unique item IDs
        for s in self.superitems_pool:
            try:
                dims = s.get_items_dims()
                for item_id, dim in dims.items():
                    if item_id not in seen_items:
                        items_dims[item_id] = dim
                        seen_items.add(item_id)
                    else:
                        logger.warning(f"Duplicate item {item_id} detected, skipping")
            except TypeError as e:
                logger.error(f"TypeError for superitem with IDs {s.id}: {e}; skipping")
        return items_dims

    def get_unique_items_ids(self):
        """Return the flattened list of item ids inside the layer."""
        return self.superitems_pool.get_unique_item_ids()

    def get_density(self, two_dims=False):
        """Compute the 2D/3D density of the layer."""
        return (
            self.volume / (self.pallet_dims.area * self.height + 1e-6)
            if not two_dims
            else self.area / self.pallet_dims.area
        )

    def remove(self, superitem):
        """Return a new layer without the given superitem."""
        new_spool = superitems.SuperitemPool(
            superitems=[s for s in self.superitems_pool if s != superitem]
        )
        new_scoords = [
            c
            for i, c in enumerate(self.superitems_coords)
            if i != self.superitems_pool.get_index(superitem)
        ]
        return Layer(new_spool, new_scoords, self.pallet_dims)

    def get_superitems_containing_item(self, item_id):
        """Return a list of superitems containing the given raw item."""
        return self.superitems_pool.get_superitems_containing_item(item_id)

    def rearrange(self):
        """Apply maxrects over superitems in layer."""
        return maxrects.maxrects_single_layer_offline(self.superitems_pool, self.pallet_dims)

    def plot(self, ax=None, height=0):
        """Plot items in the current layer in the given plot or  in a new 3D plot."""
        if ax is None:
            ax = utils.get_pallet_plot(
                utils.Dimension(self.pallet_dims.width, self.pallet_dims.length, self.height)
            )
        items_coords = self.get_items_coords(z=height)
        items_dims = self.get_items_dims()
        for item_id in items_coords.keys():
            coords = items_coords[item_id]
            dims = items_dims[item_id]
            ax = utils.plot_product(ax, item_id, coords, dims)
        return ax

    def to_dataframe(self, z=0):  # z unused parameter kept for compatibility
        items_coords = self.get_items_coords()  # No z needed
        items_dims = self.get_items_dims()
        keys = list(items_coords.keys())
        xs = [items_coords[k].x for k in keys]
        ys = [items_coords[k].y for k in keys]
        zs = [items_coords[k].z for k in keys]
        ws = [items_dims[k].width for k in keys]
        ds = [items_dims[k].length for k in keys]
        hs = [items_dims[k].height for k in keys]

        # Add weight information by looking up the superitem
        weights = []
        for key in keys:
            weight = 0.0
            for superitem in self.superitems_pool:
                if key in superitem.id:
                    for raw_item in superitem.get_items():
                        if raw_item.id == key:
                            weight = raw_item.weight
                            break
                    break
            weights.append(weight)

        return pd.DataFrame(
            {
                "item": keys,
                "x": xs,
                "y": ys,
                "z": zs,
                "width": ws,
                "length": ds,
                "height": hs,
                "weight": weights,
            }
        )

    def __str__(self):
        return f"Layer(height={self.height}, ids={self.superitems_pool.get_unique_item_ids()})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.get_items_coords() == other.get_items_coords()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.superitems_pool)

    def __contains__(self, superitem):
        return superitem in self.superitems_pool

    def __hash__(self):
        s_hashes = [hash(s) for s in self.superitems_pool]
        c_hashes = [hash(c) for c in self.superitems_coords]
        if len(s_hashes) != len(c_hashes):
            min_length = min(len(s_hashes), len(c_hashes))
            s_hashes, c_hashes = s_hashes[:min_length], c_hashes[:min_length]  # Align lengths

        strs = [f"{s_hashes[i]}/{c_hashes[i]}" for i in utils.argsort(s_hashes)]

        # strs = [f"{s_hashes[i]}/{c_hashes[i]}" for i in utils.argsort(s_hashes)]
        return hash("-".join(strs))


class LayerPool:
    """Collection of layers for multi-layer packing solutions."""

    def __init__(self, superitems_pool, pallet_dims, layers=None, add_single=False):
        """Initialize a layer pool with optional single-item layers."""
        self.superitems_pool = superitems_pool
        self.pallet_dims = pallet_dims
        self.layers = layers or []
        self.hash_to_index = self._get_hash_to_index()

        if add_single:
            self._add_single_layers()

    def _get_hash_to_index(self):
        """Create hash-to-index mapping for fast layer lookup."""
        return {hash(l): i for i, l in enumerate(self.layers)}

    def _add_single_layers(self):
        """Add individual layers for each superitem in the pool."""
        for superitem in self.superitems_pool:
            self.add(
                Layer(
                    superitems.SuperitemPool([superitem]),
                    [utils.Coordinate(x=0, y=0)],
                    self.pallet_dims,
                )
            )

    def subset(self, layer_indices):
        """Create a new layer pool with specified layers."""
        layers = [l for i, l in enumerate(self.layers) if i in layer_indices]
        return LayerPool(self.superitems_pool, self.pallet_dims, layers=layers)

    def difference(self, layer_indices):
        """Create a new layer pool excluding specified layers."""
        layers = [l for i, l in enumerate(self.layers) if i not in layer_indices]
        return LayerPool(self.superitems_pool, self.pallet_dims, layers=layers)

    def get_ol(self):
        """Return a numpy array ol s.t. ol[l] = h iff."""
        return np.array([layer.height for layer in self.layers], dtype=int)

    def get_zsl(self):
        """Return a binary matrix zsl s.t. zsl[s, l] = 1 iff."""
        zsl = np.zeros((len(self.superitems_pool), len(self.layers)), dtype=int)
        for s, superitem in enumerate(self.superitems_pool):
            for l, layer in enumerate(self.layers):
                if superitem in layer:
                    zsl[s, l] = 1
        return zsl

    def add(self, layer):
        """Add the given layer to the current pool."""
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        l_hash = hash(layer)
        if l_hash not in self.hash_to_index:
            self.layers.append(layer)
            self.hash_to_index[l_hash] = len(self.layers) - 1

    def extend(self, layer_pool):
        """Extend the current pool with the given one."""
        assert isinstance(
            layer_pool, LayerPool
        ), "The given set of layers should be an instance of the LayerPool class"
        check_dims = layer_pool.pallet_dims == self.pallet_dims
        assert check_dims, "The given LayerPool is defined over different pallet dimensions"
        for layer in layer_pool:
            self.add(layer)
        self.superitems_pool.extend(layer_pool.superitems_pool)

    def remove(self, layer):
        """Remove the given Layer from the LayerPool."""
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        l_hash = hash(layer)
        if l_hash in self.hash_to_index:
            del self.layers[self.hash_to_index[l_hash]]
            self.hash_to_index = self._get_hash_to_index()

    def replace(self, i, layer):
        """Replace layer at index i with the given layer."""
        assert i in range(len(self.layers)), "Index out of bounds"
        assert isinstance(layer, Layer), "The given layer should be an instance of the Layer class"
        del self.hash_to_index[hash(self.layers[i])]
        self.hash_to_index[hash(layer)] = i
        self.layers[i] = layer

    def pop(self, i):
        """Remove the layer at the given index from the pool."""
        self.remove(self.layers[i])

    def get_unique_items_ids(self):
        """Return the flattened list of item ids inside the layer pool."""
        return self.superitems_pool.get_unique_item_ids()

    def get_densities(self, two_dims=False):
        """Compute the 2D/3D density of each layer in the pool."""
        return [layer.get_density(two_dims=two_dims) for layer in self.layers]

    def sort_by_densities(self, two_dims=False):
        """Sort layers in the pool by decreasing density."""
        densities = self.get_densities(two_dims=two_dims)
        sorted_indices = utils.argsort(densities, reverse=True)
        self.layers = [self.layers[i] for i in sorted_indices]

    def discard_by_densities(self, min_density=0.5, two_dims=False):
        """Sort layers by densities and keep only those with a."""
        assert min_density >= 0.0, "Density tolerance must be non-negative"
        self.sort_by_densities(two_dims=two_dims)
        densities = self.get_densities(two_dims=two_dims)
        last_index = -1
        for i, d in enumerate(densities):
            if d >= min_density:
                last_index = i
            else:
                break
        return self.subset(list(range(last_index + 1)))

    def discard_by_coverage(self, max_coverage_all=3, max_coverage_single=3):
        """Post-process layers by their item coverage."""
        assert max_coverage_all > 0, "Maximum number of covered items in all layers must be > 0"
        assert (
            max_coverage_single > 0
        ), "Maximum number of covered items in a single layer must be > 0"
        all_item_ids = self.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [0] * len(all_item_ids)))
        layers_to_select = []
        for l, layer in enumerate(self.layers):
            to_select = True
            already_covered = 0

            # Stop when all items are covered
            if all([c > 0 for c in item_coverage.values()]):
                break

            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                # If at least one item in the layer was already selected
                # more times than the maximum allowed value, then such layer
                # is to be discarded
                if item_coverage[item] >= max_coverage_all:
                    to_select = False
                    break

                # If at least `max_coverage_single` items in the layer are already covered
                # by previously selected layers, then such layer is to be discarded
                if item_coverage[item] > 0:
                    already_covered += 1
                if already_covered >= max_coverage_single:
                    to_select = False
                    break

            # If the layer is selected, increase item coverage
            # for each item in such layer and add it to the pool
            # of selected layers
            if to_select:
                layers_to_select += [l]
                for item in item_ids:
                    item_coverage[item] += 1

        return self.subset(layers_to_select)

    def remove_duplicated_items(self, min_density=0.5, two_dims=False):
        """Keep items that are covered multiple times only."""
        assert min_density >= 0.0, "Density tolerance must be non-negative"
        selected_layers = copy.deepcopy(self)
        all_item_ids = selected_layers.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        edited_layers, to_remove = set(), set()
        for l in range(len(selected_layers)):
            layer = selected_layers[l]
            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                duplicated_superitems, duplicated_indices = layer.get_superitems_containing_item(
                    item
                )
                # Remove superitems in different layers containing the same item
                # (remove the ones in less dense layers)
                if item_coverage[item]:
                    edited_layers.add(l)
                    layer = layer.difference(duplicated_indices)
                    logger.debug(f"Removing duplicated items in layer {l} containing item {item}")
                # Remove superitems in the same layer containing the same item
                # (remove the ones with less volume)
                elif len(duplicated_indices) > 1:
                    edited_layers.add(l)
                    duplicated_volumes = [s.volume for s in duplicated_superitems]
                    layer = layer.difference(
                        [duplicated_indices[i] for i in utils.argsort(duplicated_volumes)[:-1]]
                    )

            if l in edited_layers:
                # Flag the layer if it doesn't respect the minimum density
                density = layer.get_density(two_dims=two_dims)
                if density < min_density or density == 0:
                    to_remove.add(l)
                # Replace the original layer with the edited one
                else:
                    selected_layers.replace(l, layer)

            # Update item coverage
            if l not in to_remove:
                item_ids = selected_layers[l].get_unique_items_ids()
                for item in item_ids:
                    item_coverage[item] = True

        # Rearrange layers in which at least one superitem was removed
        for l in edited_layers:
            if l not in to_remove:
                layer = selected_layers[l].rearrange()
                if layer is not None:
                    selected_layers[l] = layer
                else:
                    logger.error(f"After removing duplicated items couldn't rearrange layer {l}")

        # Removing layers last to first to avoid indexing errors
        for l in sorted(to_remove, reverse=True):
            selected_layers.pop(l)

        return selected_layers

    def remove_empty_layers(self):
        """Check and remove layers without any items."""
        not_empty_layers = []
        for l, layer in enumerate(self.layers):
            if not layer.is_empty():
                not_empty_layers.append(l)
        return self.subset(not_empty_layers)

    def filter_layers(
        self,
        min_density=0.5,
        two_dims=False,
        max_coverage_all=3,
        max_coverage_single=3,
        visualize_filtered_layers=False,
    ):
        """Perform post-processing steps to select the best layers in the pool."""

        logger.info(f"Filtering {len(self)} generated layers")
        new_pool = self.discard_by_densities(min_density=min_density, two_dims=two_dims)
        logger.debug(f"Remaining {len(new_pool)} layers after discarding by {min_density} density")
        new_pool = new_pool.discard_by_coverage(
            max_coverage_all=max_coverage_all, max_coverage_single=max_coverage_single
        )
        logger.debug(
            f"Remaining {len(new_pool)} layers after discarding by coverage "
            f"(all: {max_coverage_all}, single: {max_coverage_single})"
        )
        new_pool = new_pool.remove_duplicated_items(min_density=min_density, two_dims=two_dims)
        logger.debug(f"Remaining {len(new_pool)} layers after removing duplicated items ")
        new_pool = new_pool.remove_empty_layers()
        logger.debug(f"Remaining {len(new_pool)} layers after removing the empty ones")
        new_pool.sort_by_densities(two_dims=two_dims)

        if visualize_filtered_layers:
            # Visualization before and after filtering
            visualization.visualize_pre_post_filter(self, new_pool, self.pallet_dims)

        return new_pool

    def item_coverage(self):
        """Return a dictionary {i: T/F} identifying whether or not."""
        all_item_ids = self.get_unique_items_ids()
        item_coverage = dict(zip(all_item_ids, [False] * len(all_item_ids)))
        for layer in self.layers:
            item_ids = layer.get_unique_items_ids()
            for item in item_ids:
                item_coverage[item] = True

        return item_coverage

    def not_covered_single_superitems(self, singles_removed=None):
        """Return a list of single item superitems that are not present in the pool."""
        # Get items not covered in the layer pool
        item_coverage = self.item_coverage()
        not_covered_ids = [k for k, v in item_coverage.items() if not v]
        not_covered = set()
        for s in self.superitems_pool:
            for i in not_covered_ids:
                if s.id == [i]:
                    not_covered.add(s)

        # Add not covered single items that were removed due to
        # layer filtering of horizontal superitems
        singles_removed = singles_removed or []
        for s in singles_removed:
            if s.id[0] not in item_coverage:
                not_covered.add(s)

        return list(not_covered)

    def not_covered_superitems(self):
        """Return a list of superitems which are not present in any layer."""
        covered_spool = superitems.SuperitemPool(superitems=None)
        for l in self.layers:
            covered_spool.extend(l.superitems_pool)

        return [s for s in self.superitems_pool if covered_spool.get_index(s) is None]

    def get_heights(self):
        """Return the list of layer heights in the pool."""
        return [l.height for l in self.layers]

    def get_areas(self):
        """Return the list of layer areas in the pool."""
        return [l.area for l in self.layers]

    def get_volumes(self):
        """Return the list of layer volumes in the pool."""
        return [l.volume for l in self.layers]

    def to_dataframe(self, zs=None):
        """Convert the layer pool to a Pandas DataFrame."""
        if len(self) == 0:
            return pd.DataFrame()
        if zs is None:
            zs = [0] * len(self)
        dfs = []
        for i, layer in enumerate(self.layers):
            df = layer.to_dataframe(z=zs[i])
            df["layer"] = [i] * len(df)
            dfs += [df]
        return pd.concat(dfs, axis=0).reset_index(drop=True)

    def describe(self):
        """Return a DataFrame with stats about the current layer pool."""
        ids = list(range(len(self.layers)))
        heights = self.get_heights()
        areas = self.get_areas()
        volumes = self.get_volumes()
        densities_2d = self.get_densities(two_dims=True)
        densities_3d = self.get_densities(two_dims=False)
        df = pd.DataFrame(
            zip(ids, heights, areas, volumes, densities_2d, densities_3d),
            columns=["layer", "height", "area", "volume", "2d_density", "3d_density"],
        )
        total = (
            df.agg(
                {
                    "height": np.sum,
                    "area": np.sum,
                    "volume": np.sum,
                    "2d_density": np.mean,
                    "3d_density": np.mean,
                }
            )
            .to_frame()
            .T
        )
        total["layer"] = "Total"
        return pd.concat((df, total), axis=0).reset_index(drop=True)

    def __str__(self):
        return f"LayerPool(layers={self.layers})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.layers)

    def __contains__(self, layer):
        return layer in self.layers

    def __getitem__(self, i):
        return self.layers[i]

    def __setitem__(self, i, e):
        assert isinstance(e, Layer), "The given layer should be an instance of the Layer class"
        self.layers[i] = e
