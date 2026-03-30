import itertools
import time
from collections import Counter
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

from src.models import layers, superitems


class Dimension:
    """Store the dimensions and weight of a 3D object."""

    def __init__(self, width, length, height, weight=0):
        """Initialize dimensions and derived geometry values."""
        self.width = int(width)
        self.length = int(length)
        self.height = int(height)
        self.weight = float(weight)
        self.area = int(width * length)
        self.volume = int(width * length * height)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.width == other.width
                and self.length == other.length
                and self.height == other.height
                and self.weight == other.weight
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            f"Dimension(width={self.width}, length={self.length}, height={self.height}, "
            f"weight={self.weight}, volume={self.volume})"
        )

    def __repr__(self):
        return self.__str__()


class Coordinate:
    """Represent a 3D coordinate using the bottom-left-back convention."""

    def __init__(self, x, y, z=0):
        """Initialize a 3D coordinate."""
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def from_blb_to_vertices(self, dims):
        """Return the eight vertices of a cuboid from its base corner."""
        assert isinstance(dims, Dimension), "The given dimension should be an instance of Dimension"
        blb = self
        blf = Coordinate(self.x + dims.width, self.y, self.z)
        brb = Coordinate(self.x, self.y + dims.length, self.z)
        brf = Coordinate(self.x + dims.width, self.y + dims.length, self.z)
        tlb = Coordinate(self.x, self.y, self.z + dims.height)
        tlf = Coordinate(self.x + dims.width, self.y, self.z + dims.height)
        trb = Coordinate(self.x, self.y + dims.length, self.z + dims.height)
        trf = Coordinate(self.x + dims.width, self.y + dims.length, self.z + dims.height)
        return [blb, blf, brb, brf, tlb, tlf, trb, trf]

    def to_numpy(self):
        """Convert coordinates to numpy array."""
        return np.array([self.x, self.y, self.z])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))


class Vertices:
    """Represent the eight vertices of a 3D cuboid."""

    def __init__(self, blb, dims):
        assert isinstance(
            blb, Coordinate
        ), "The given coordinate should be an instance of Coordinate"
        assert isinstance(dims, Dimension), "The given dimension should be an instance of Dimension"
        self.dims = dims

        # Bottom left back and front
        self.blb = blb
        self.blf = Coordinate(self.blb.x + self.dims.width, self.blb.y, self.blb.z)

        # Bottom right back and front
        self.brb = Coordinate(self.blb.x, self.blb.y + self.dims.length, self.blb.z)
        self.brf = Coordinate(
            self.blb.x + self.dims.width, self.blb.y + self.dims.length, self.blb.z
        )

        # Top left back and front
        self.tlb = Coordinate(self.blb.x, self.blb.y, self.blb.z + self.dims.height)
        self.tlf = Coordinate(
            self.blb.x + self.dims.width, self.blb.y, self.blb.z + self.dims.height
        )

        # Top right back and front
        self.trb = Coordinate(
            self.blb.x, self.blb.y + self.dims.length, self.blb.z + self.dims.height
        )
        self.trf = Coordinate(
            self.blb.x + self.dims.width,
            self.blb.y + self.dims.length,
            self.blb.z + self.dims.height,
        )

        # List of vertices
        self.vertices = [
            self.blb,
            self.blf,
            self.brb,
            self.brf,
            self.tlb,
            self.tlf,
            self.trb,
            self.trf,
        ]

    def get_center(self):
        """Return the central coordinate of the cuboid."""
        return Coordinate(
            self.blb.x + self.dims.width // 2,
            self.blb.y + self.dims.length // 2,
            self.blb.z + self.dims.height // 2,
        )

    def get_xs(self):
        """Return a numpy array containing all the x-values."""
        return np.array([v.x for v in self.vertices])

    def get_ys(self):
        """Return a numpy array containing all the y-values."""
        return np.array([v.y for v in self.vertices])

    def get_zs(self):
        """Return a numpy array containing all the z-values."""
        return np.array([v.z for v in self.vertices])

    def to_faces(self):
        """Convert the computed set of vertices to a list of faces."""
        return np.array(
            [
                [
                    self.blb.to_numpy(),
                    self.blf.to_numpy(),
                    self.brf.to_numpy(),
                    self.brb.to_numpy(),
                ],  # bottom
                [
                    self.tlb.to_numpy(),
                    self.tlf.to_numpy(),
                    self.trf.to_numpy(),
                    self.trb.to_numpy(),
                ],  # top
                [
                    self.blb.to_numpy(),
                    self.brb.to_numpy(),
                    self.trb.to_numpy(),
                    self.tlb.to_numpy(),
                ],  # back
                [
                    self.blf.to_numpy(),
                    self.brf.to_numpy(),
                    self.trf.to_numpy(),
                    self.tlf.to_numpy(),
                ],  # front
                [
                    self.blb.to_numpy(),
                    self.blf.to_numpy(),
                    self.tlf.to_numpy(),
                    self.tlb.to_numpy(),
                ],  # left
                [
                    self.brb.to_numpy(),
                    self.brf.to_numpy(),
                    self.trf.to_numpy(),
                    self.trb.to_numpy(),
                ],  # right
            ]
        )


def argsort(seq, reverse=False):
    """Sort the given array and return indices instead of values."""
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def duplicate_keys(dicts):
    """Check that the input dictionaries have common keys."""
    keys = list(flatten([d.keys() for d in dicts]))
    return [k for k, v in Counter(keys).items() if v > 1]


def flatten(l):
    """Recursively flatten nested lists into a single-level generator."""
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def build_layer_from_model_output(superitems_pool, superitems_in_layer, solution, pallet_dims):
    """Return a single layer from the given model solution (either baseline or column generation)."""
    spool, scoords = [], []
    for s in superitems_in_layer:
        spool += [superitems_pool[s]]
        scoords += [Coordinate(x=solution[f"c_{s}_x"], y=solution[f"c_{s}_y"])]
    spool = superitems.SuperitemPool(superitems=spool)
    return layers.Layer(spool, scoords, pallet_dims)


def do_overlap(a, b):
    """Check if two items (Pandas Series) overlap."""
    assert isinstance(a, pd.Series) and isinstance(b, pd.Series), "Wrong input types"
    dx = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    dy = min(a.y + a.length, b.y + b.length) - max(a.y, b.y)
    dz = min(a.z + a.height, b.z + b.height) - max(a.z, b.z)
    if dx > 0 and dy > 0 and dz > 0:
        return True
    return False


def get_l0_lb(order, pallet_dims):
    """Return the continuous lower bound on the number of bins."""
    return np.ceil(order.volume.sum() / pallet_dims.volume)


def get_l1_lb(order, pallet_dims):
    """L1 lower bound for the minimum number of required bins."""

    def get_j2(d1, bd1, d2, bd2):
        return order[(order[d1] > (bd1 / 2)) & (order[d2] > (bd2 / 2))]

    def get_js(j2, p, d, bd):
        return j2[(j2[d] >= p) & (j2[d] <= (bd / 2))]

    def get_jl(j2, p, d, bd):
        return j2[(j2[d] > (bd / 2)) & (j2[d] <= bd - p)]

    def get_l1j2(d1, bd1, d2, bd2, d3, bd3):
        j2 = get_j2(d1, bd1, d2, bd2)
        if len(j2) == 0:
            return 0.0
        ps = order[order[d3] <= bd3 / 2][d3].values
        max_ab = -np.inf
        for p in tqdm(ps):
            js = get_js(j2, p, d3, bd3)
            jl = get_jl(j2, p, d3, bd3)
            a = np.ceil((js[d3].sum() - (len(jl) * bd3 - jl[d3].sum())) / bd3)
            b = np.ceil((len(js) - (np.floor((bd3 - jl[d3].values) / p)).sum()) / np.floor(bd3 / p))
            max_ab = max(max_ab, a, b)

        return len(j2[j2[d3] > (bd3 / 2)]) + max_ab

    l1wh = get_l1j2(
        "width", pallet_dims.width, "height", pallet_dims.height, "length", pallet_dims.length
    )
    l1wd = get_l1j2(
        "width", pallet_dims.width, "length", pallet_dims.length, "height", pallet_dims.height
    )
    l1dh = get_l1j2(
        "length", pallet_dims.length, "width", pallet_dims.width, "height", pallet_dims.height
    )
    return max(l1wh, l1wd, l1dh), l1wh, l1wd, l1dh


def get_l2_lb(order, pallet_dims):
    """L2 lower bound for the minimum number of required bins."""

    def get_kv(p, q, d1, bd1, d2, bd2):
        return order[(order[d1] > bd1 - p) & (order[d2] > bd2 - q)]

    def get_kl(kv, d1, bd1, d2, bd2):
        kl = order[~order.isin(kv)]
        return kl[(kl[d1] > (bd1 / 2)) & (kl[d2] > (bd2 / 2))]

    def get_ks(kv, kl, p, q, d1, d2):
        ks = order[~order.isin(pd.concat([kv, kl], axis=0))]
        return ks[(ks[d1] >= p) & (ks[d2] >= q)]

    def get_l2j2pq(p, q, l1, d1, bd1, d2, bd2, d3, bd3):
        kv = get_kv(p, q, d1, bd1, d2, bd2)
        kl = get_kl(kv, d1, bd1, d2, bd2)
        ks = get_ks(kv, kl, p, q, d1, d2)

        return l1 + max(
            0,
            np.ceil(
                (pd.concat([kl, ks], axis=0).volume.sum() - (bd3 * l1 - kv[d3].sum()) * bd1 * bd2)
                / (bd1 * bd2 * bd3)
            ),
        )

    def get_l2j2(l1, d1, bd1, d2, bd2, d3, bd3):
        ps = order[(order[d1] <= bd1 // 2)][d1].values
        qs = order[(order[d2] <= bd2 // 2)][d2].values
        max_l2j2 = -np.inf
        for p, q in tqdm(itertools.product(ps, qs)):
            l2j2 = get_l2j2pq(p, q, l1, d1, bd1, d2, bd2, d3, bd3)
            max_l2j2 = max(max_l2j2, l2j2)
        return max_l2j2

    _, l1wh, l1wd, l1hd = get_l1_lb(order, pallet_dims)
    l2wh = get_l2j2(
        l1wh, "width", pallet_dims.width, "height", pallet_dims.height, "length", pallet_dims.length
    )
    l2wd = get_l2j2(
        l1wd, "width", pallet_dims.width, "length", pallet_dims.length, "height", pallet_dims.height
    )
    l2dh = get_l2j2(
        l1hd, "length", pallet_dims.length, "height", pallet_dims.height, "width", pallet_dims.width
    )
    return max(l2wh, l2wd, l2dh), l2wh, l2wd, l2dh


def get_height_groups(superitems_pool, pallet_dims, height_tol=0, density_tol=0.5):
    """Divide the whole pool of superitems into groups having either."""
    assert height_tol >= 0 and density_tol >= 0.0, "Tolerance parameters must be non-negative"

    # Get unique heights
    unique_heights = sorted(set(s.height for s in superitems_pool))
    height_sets = {
        h: {k for k in unique_heights[i:] if k - h <= height_tol}
        for i, h in enumerate(unique_heights)
    }
    for (i, hi), (j, hj) in zip(list(height_sets.items())[:-1], list(height_sets.items())[1:]):
        if hj.issubset(hi):
            unique_heights.remove(j)

    # Generate one group of superitems for each similar height
    groups = []
    for height in unique_heights:
        spool = [
            s for s in superitems_pool if s.height >= height and s.height <= height + height_tol
        ]
        spool = superitems.SuperitemPool(superitems=spool)
        if (
            sum(s.volume for s in spool)
            >= density_tol * spool.get_max_height() * pallet_dims.width * pallet_dims.length
        ):
            groups += [spool]

    return groups


# Add or update this function in utils.py


def calculate_bin_statistics(compact_bin_pool, PALLET_DIMS=None, time_elapsed=0):
    """Calculate statistics for a compact bin pool with correct height calculation."""
    import pandas as pd

    # Check if compact_bin_pool is empty
    if not compact_bin_pool or len(compact_bin_pool.compact_bins) == 0:
        return pd.DataFrame()

    # Initialize statistics dictionary
    stats = {
        "Bin": [],
        "Original_Utilization_Percent": [],
        "Compact_Utilization_Percent": [],
        "Item_Count": [],
        "Max_Height": [],  # Add this to track max height
        "Max_Height_Percent": [],  # Add this to track height utilization
    }

    # Get the dataframe representation of the compact bin pool
    df = compact_bin_pool.to_dataframe()

    # Calculate statistics for each bin
    for bin_idx, bin in enumerate(compact_bin_pool.compact_bins):
        # Filter data for this bin
        bin_df = df[df["bin"] == bin_idx]

        # Calculate item count
        item_count = len(bin_df)

        # Calculate total item volume
        item_volume = sum(bin_df["width"] * bin_df["length"] * bin_df["height"])

        # Calculate bin volume based on pallet dimensions
        bin_volume = (
            PALLET_DIMS.volume if PALLET_DIMS else (1200 * 800 * 1850)
        )  # Default if not provided

        # Calculate utilization percentage
        utilization_percent = (item_volume / bin_volume) * 100

        # Calculate maximum height (z + height of the highest item)
        if not bin_df.empty:
            # Ensure numeric values
            for col in ["z", "height"]:
                bin_df[col] = pd.to_numeric(bin_df[col])

            # Calculate each item's top z-coordinate
            bin_df["top_z"] = bin_df["z"] + bin_df["height"]

            # Find maximum height
            max_height = bin_df["top_z"].max()
            max_height_percent = (max_height / PALLET_DIMS.height) * 100 if PALLET_DIMS else 0
        else:
            max_height = 0
            max_height_percent = 0

        # Add statistics to our dictionary
        stats["Bin"].append(bin_idx + 1)
        stats["Original_Utilization_Percent"].append(utilization_percent)
        stats["Compact_Utilization_Percent"].append(utilization_percent)  # Same for compact bins
        stats["Item_Count"].append(item_count)
        stats["Max_Height"].append(max_height)
        stats["Max_Height_Percent"].append(max_height_percent)

    # Create a DataFrame from the statistics
    stats_df = pd.DataFrame(stats)

    # Calculate averages
    avg_orig_util = stats_df["Original_Utilization_Percent"].mean()
    avg_compact_util = stats_df["Compact_Utilization_Percent"].mean()
    avg_items = stats_df["Item_Count"].mean()
    total_bins = len(stats_df)
    total_items = stats_df["Item_Count"].sum()
    total_layers = sum(bin.df["layer"].nunique() for bin in compact_bin_pool.compact_bins)
    avg_max_height = stats_df["Max_Height"].mean()
    avg_height_percent = stats_df["Max_Height_Percent"].mean()

    # Create a summary row
    summary = pd.DataFrame(
        {
            "Bin": ["Average"],
            "Original_Utilization_Percent": [avg_orig_util],
            "Compact_Utilization_Percent": [avg_compact_util],
            "Item_Count": [avg_items],
            "Max_Height": [avg_max_height],
            "Max_Height_Percent": [avg_height_percent],
            "Total_Bins": [total_bins],
            "Total_Items": [total_items],
            "Total_Layers": [total_layers],
            "Time_Elapsed": [time_elapsed],
        }
    )

    # Add the summary row to the statistics DataFrame
    stats_df = pd.concat([stats_df, summary], ignore_index=True)

    return stats_df


def calculate_overlap_area(rect1, rect2):
    """Calculate the 2D overlap area between two rectangles."""
    x1, y1, w1, l1 = rect1
    x2, y2, w2, l2 = rect2

    # Calculate the overlap in x and y directions
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + l1, y2 + l2) - max(y1, y2))

    # Return the area of overlap
    return x_overlap * y_overlap


def calculate_support_percentage(item_rect, support_rects):
    """Calculate the percentage of an item's base area that is supported."""
    x, y, w, l = item_rect
    item_area = w * l

    if item_area == 0:
        return 0  # Avoid division by zero

    # Calculate total supported area
    total_support = 0
    for support_rect in support_rects:
        overlap = calculate_overlap_area(item_rect, support_rect)
        total_support += overlap

    # Ensure we don't count overlapping support areas twice
    total_support = min(total_support, item_area)

    # Return the percentage of supported area
    return (total_support / item_area) * 100


# Add this function to utils.py


def get_adaptive_support_requirement(
    item_weight=None,
    item_height=None,
    z_level=0,
    default_support=70.0,
    min_support=50.0,
    max_support=90.0,
):
    """Calculate an adaptive stability requirement based on item properties."""
    # Start with default requirement
    required_support = default_support

    # Adjust based on height from floor (higher items need more support)
    if z_level is not None:
        height_factor = min(1.0, z_level / 1000.0)  # Normalize to 1.0 at z=1000
        required_support += height_factor * 10.0  # Add up to 10% more for higher items

    # Adjust based on item weight if available (heavier items need more support)
    if item_weight is not None and item_weight > 0:
        # Assuming weights typically range from 0-50kg, normalize to 0-1
        weight_factor = min(1.0, item_weight / 50.0)
        required_support += weight_factor * 15.0  # Add up to 15% more for heavier items

    # Adjust based on item height (taller items need more support)
    if item_height is not None and item_height > 0:
        # Taller items are less stable
        height_shape_factor = min(1.0, item_height / 200.0)  # Normalize to 1.0 at height=200
        required_support += height_shape_factor * 5.0  # Add up to 5% more for taller items

    # Ensure the requirement stays within bounds
    required_support = max(min_support, min(max_support, required_support))

    return required_support


# Then modify the has_sufficient_support function to use adaptive requirements


def has_sufficient_support(
    item_rect, support_rects, min_support_pct=70, item_weight=None, item_height=None, z_level=0
):
    """Check if an item has sufficient support using adaptive requirements."""
    # Calculate adaptive requirement
    required_support = get_adaptive_support_requirement(
        item_weight=item_weight,
        item_height=item_height,
        z_level=z_level,
        default_support=min_support_pct,
    )

    # Calculate actual support
    support_pct = calculate_support_percentage(item_rect, support_rects)

    # Items on the floor (z=0) are always stable
    if z_level == 0:
        return True

    return support_pct >= required_support


# Add this function to your utils.py file


def save_json(data, filename):
    """Save data to JSON file with NumPy type serialization support."""
    import json

    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        """Custom encoder for NumPy data types."""

        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
