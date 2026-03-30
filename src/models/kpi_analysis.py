from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BinDimensions:
    width: float
    length: float
    height: float

    @property
    def volume(self):
        return self.width * self.length * self.height


class HeightWidthRatio:
    """Class for calculating the HeightWidthRatio KPI."""

    def __init__(self, bin_dims: BinDimensions):
        """Initialize with bin dimensions."""
        self.bin_dims = bin_dims

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the HeightWidthRatio KPI."""
        if items_df.empty:
            return 0.0

        # Calculate height-to-base ratio for each item
        items_df = items_df.copy()
        items_df["base_area"] = items_df["width"] * items_df["length"]
        items_df["height_base_ratio"] = items_df["height"] / np.sqrt(items_df["base_area"])

        # Normalize ratios to identify relatively tall items
        max_ratio = items_df["height_base_ratio"].max()
        if max_ratio > 0:
            items_df["normalized_ratio"] = items_df["height_base_ratio"] / max_ratio
        else:
            items_df["normalized_ratio"] = 0

        # Calculate distance from center of bin (in xy plane)
        center_x = self.bin_dims.width / 2
        center_y = self.bin_dims.length / 2
        items_df["center_x"] = items_df["x"] + items_df["width"] / 2
        items_df["center_y"] = items_df["y"] + items_df["length"] / 2

        items_df["dist_from_center"] = np.sqrt(
            (items_df["center_x"] - center_x) ** 2 + (items_df["center_y"] - center_y) ** 2
        )

        # Normalize distance (0 = center, 1 = furthest possible corner)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        items_df["normalized_dist"] = items_df["dist_from_center"] / max_dist if max_dist > 0 else 0

        # Calculate normalized height (0 = bottom, 1 = top)
        items_df["item_mid_z"] = items_df["z"] + items_df["height"] / 2
        items_df["normalized_height"] = (
            items_df["item_mid_z"] / self.bin_dims.height if self.bin_dims.height > 0 else 0
        )

        # Calculate stability score for each item
        # Lower scores for tall items near edges or top
        items_df["item_stability"] = 1 - (
            items_df["normalized_ratio"]
            * (0.7 * items_df["normalized_dist"] + 0.3 * items_df["normalized_height"])
        )

        # Weight by volume to give more importance to larger items
        items_df["volume"] = items_df["width"] * items_df["length"] * items_df["height"]
        total_volume = items_df["volume"].sum()

        if total_volume == 0:
            return 0.0

        weighted_score = np.sum(items_df["item_stability"] * items_df["volume"]) / total_volume

        # Return normalized score between 0 and 1
        return max(0.0, min(1.0, weighted_score))

    def get_item_scores(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Return individual stability scores for each item."""
        if items_df.empty:
            return pd.DataFrame(columns=["item", "height_base_ratio", "stability_score"])

        # Calculate using same logic as in calculate()
        items_df = items_df.copy()
        items_df["base_area"] = items_df["width"] * items_df["length"]
        items_df["height_base_ratio"] = items_df["height"] / np.sqrt(items_df["base_area"])

        max_ratio = items_df["height_base_ratio"].max()
        if max_ratio > 0:
            items_df["normalized_ratio"] = items_df["height_base_ratio"] / max_ratio
        else:
            items_df["normalized_ratio"] = 0

        center_x = self.bin_dims.width / 2
        center_y = self.bin_dims.length / 2
        items_df["center_x"] = items_df["x"] + items_df["width"] / 2
        items_df["center_y"] = items_df["y"] + items_df["length"] / 2

        items_df["dist_from_center"] = np.sqrt(
            (items_df["center_x"] - center_x) ** 2 + (items_df["center_y"] - center_y) ** 2
        )

        max_dist = np.sqrt(center_x**2 + center_y**2)
        items_df["normalized_dist"] = items_df["dist_from_center"] / max_dist if max_dist > 0 else 0

        items_df["item_mid_z"] = items_df["z"] + items_df["height"] / 2
        items_df["normalized_height"] = (
            items_df["item_mid_z"] / self.bin_dims.height if self.bin_dims.height > 0 else 0
        )

        items_df["stability_score"] = 1 - (
            items_df["normalized_ratio"]
            * (0.7 * items_df["normalized_dist"] + 0.3 * items_df["normalized_height"])
        )

        return items_df[["item", "height_base_ratio", "stability_score"]]


class RelativeDensity:
    """Class for calculating the Relative Density KPI."""

    def __init__(self, bin_dims: BinDimensions, method: str = "bounding_box"):
        """Initialize with bin dimensions and calculation method."""
        self.bin_dims = bin_dims
        self.method = method

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the Relative Density KPI."""
        if items_df.empty:
            return 0.0

        # Calculate total volume of all packed items
        items_df = items_df.copy()
        items_df["volume"] = items_df["width"] * items_df["length"] * items_df["height"]
        total_item_volume = items_df["volume"].sum()

        if total_item_volume == 0:
            return 0.0

        # Calculate utilized space volume based on method
        if self.method == "bounding_box":
            utilized_volume = self._calculate_bounding_box_volume(items_df)
        elif self.method == "convex_hull":
            utilized_volume = self._calculate_convex_hull_volume(items_df)
        else:
            raise ValueError("Method must be 'bounding_box' or 'convex_hull'")

        if utilized_volume == 0:
            return 0.0

        # Relative density is the ratio of item volume to utilized volume
        density = total_item_volume / utilized_volume

        # Ensure result is between 0 and 1 (should naturally be, but safety check)
        return min(1.0, max(0.0, density))

    def _calculate_bounding_box_volume(self, items_df: pd.DataFrame) -> float:
        """Calculate the volume of the axis-aligned bounding box containing all items."""
        # Find the bounds of all packed items
        min_x = items_df["x"].min()
        max_x = (items_df["x"] + items_df["width"]).max()

        min_y = items_df["y"].min()
        max_y = (items_df["y"] + items_df["length"]).max()

        min_z = items_df["z"].min()
        max_z = (items_df["z"] + items_df["height"]).max()

        # Calculate bounding box volume
        width = max_x - min_x
        length = max_y - min_y
        height = max_z - min_z

        return width * length * height

    def _calculate_convex_hull_volume(self, items_df: pd.DataFrame) -> float:
        """Calculate the volume of the 3D convex hull of all item corners."""
        # Collect all corner points of all items
        points = []

        for _, item in items_df.iterrows():
            x, y, z = item["x"], item["y"], item["z"]
            w, l, h = item["width"], item["length"], item["height"]

            # Add all 8 corners of each item
            corners = [
                [x, y, z],
                [x + w, y, z],
                [x, y + l, z],
                [x, y, z + h],
                [x + w, y + l, z],
                [x + w, y, z + h],
                [x, y + l, z + h],
                [x + w, y + l, z + h],
            ]
            points.extend(corners)

        points = np.array(points)

        # For simplicity, use bounding box of all points
        # In production, you'd want to use actual convex hull calculation
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)

        dimensions = max_coords - min_coords
        return np.prod(dimensions)

    def calculate_with_details(self, items_df: pd.DataFrame) -> dict:
        """Calculate relative density with detailed breakdown."""
        if items_df.empty:
            return {
                "density": 0.0,
                "total_item_volume": 0.0,
                "utilized_volume": 0.0,
                "waste_volume": 0.0,
                "waste_percentage": 0.0,
            }

        # Calculate volumes
        items_df = items_df.copy()
        items_df["volume"] = items_df["width"] * items_df["length"] * items_df["height"]
        total_item_volume = items_df["volume"].sum()

        if self.method == "bounding_box":
            utilized_volume = self._calculate_bounding_box_volume(items_df)
        else:
            utilized_volume = self._calculate_convex_hull_volume(items_df)

        waste_volume = max(0, utilized_volume - total_item_volume)
        waste_percentage = (waste_volume / utilized_volume * 100) if utilized_volume > 0 else 0
        density = total_item_volume / utilized_volume if utilized_volume > 0 else 0

        return {
            "density": min(1.0, max(0.0, density)),
            "total_item_volume": total_item_volume,
            "utilized_volume": utilized_volume,
            "waste_volume": waste_volume,
            "waste_percentage": waste_percentage,
            "method": self.method,
        }

    def visualize_holes(self, items_df: pd.DataFrame) -> plt.Figure:
        """Create a 3D visualization showing packed items and utilized space."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        if items_df.empty:
            ax.set_title("No items to visualize")
            return fig

        # Plot items as boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(items_df)))

        for i, (_, item) in enumerate(items_df.iterrows()):
            x, y, z = item["x"], item["y"], item["z"]
            w, l, h = item["width"], item["length"], item["height"]

            # Create box vertices - simplified wireframe representation
            # Bottom face
            ax.plot(
                [x, x + w, x + w, x, x], [y, y, y + l, y + l, y], [z, z, z, z, z], "b-", alpha=0.6
            )
            # Top face
            ax.plot(
                [x, x + w, x + w, x, x],
                [y, y, y + l, y + l, y],
                [z + h, z + h, z + h, z + h, z + h],
                "b-",
                alpha=0.6,
            )
            # Vertical edges
            for px, py in [(x, y), (x + w, y), (x + w, y + l), (x, y + l)]:
                ax.plot([px, px], [py, py], [z, z + h], "b-", alpha=0.6)

            # Add item center point
            ax.scatter([x + w / 2], [y + l / 2], [z + h / 2], c=[colors[i]], s=50, alpha=0.8)

        # Draw bounding box of utilized space
        if self.method == "bounding_box":
            min_x, max_x = items_df["x"].min(), (items_df["x"] + items_df["width"]).max()
            min_y, max_y = items_df["y"].min(), (items_df["y"] + items_df["length"]).max()
            min_z, max_z = items_df["z"].min(), (items_df["z"] + items_df["height"]).max()

            # Draw bounding box outline
            bbox_edges = [
                ([min_x, max_x], [min_y, min_y], [min_z, min_z]),
                ([min_x, max_x], [max_y, max_y], [min_z, min_z]),
                ([min_x, max_x], [min_y, min_y], [max_z, max_z]),
                ([min_x, max_x], [max_y, max_y], [max_z, max_z]),
                ([min_x, min_x], [min_y, max_y], [min_z, min_z]),
                ([max_x, max_x], [min_y, max_y], [min_z, min_z]),
                ([min_x, min_x], [min_y, max_y], [max_z, max_z]),
                ([max_x, max_x], [min_y, max_y], [max_z, max_z]),
                ([min_x, min_x], [min_y, min_y], [min_z, max_z]),
                ([max_x, max_x], [min_y, min_y], [min_z, max_z]),
                ([min_x, min_x], [max_y, max_y], [min_z, max_z]),
                ([max_x, max_x], [max_y, max_y], [min_z, max_z]),
            ]

            for edge in bbox_edges:
                ax.plot(edge[0], edge[1], edge[2], "r--", alpha=0.5, linewidth=2)

        # Calculate and display metrics
        details = self.calculate_with_details(items_df)

        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_zlabel("Height")
        ax.set_title(
            f"Packing Density Visualization\n"
            f'Density: {details["density"]:.3f} '
            f'({details["method"].replace("_", " ").title()})'
        )

        # Add text with metrics
        info_text = (
            f'Items Volume: {details["total_item_volume"]:.1f}\n'
            f'Utilized Volume: {details["utilized_volume"]:.1f}\n'
            f'Waste: {details["waste_percentage"]:.1f}%'
        )

        ax.text2D(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        return fig


class SideSupport:
    """Class for calculating the Side Support KPI."""

    def __init__(self, bin_dims: BinDimensions, min_overlap_ratio: float = 0.2):
        """Initialize with bin dimensions and minimum overlap ratio."""
        self.bin_dims = bin_dims
        self.min_overlap_ratio = min_overlap_ratio

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the Side Support KPI."""
        if items_df.empty:
            return 0.0

        total_sides = 0
        supported_sides = 0

        # For each item, check all 4 sides (excluding top and bottom)
        for i, item in items_df.iterrows():
            # Calculate item coordinates for all 6 faces
            x1, y1, z1 = item["x"], item["y"], item["z"]
            x2, y2, z2 = x1 + item["width"], y1 + item["length"], z1 + item["height"]

            # Define the 4 sides of the item as (axis, position, start1, end1, start2, end2)
            sides = [
                # Left side face (x-axis, left)
                ("x", x1, y1, y2, z1, z2),
                # Right side face (x-axis, right)
                ("x", x2, y1, y2, z1, z2),
                # Front side face (y-axis, front)
                ("y", y1, x1, x2, z1, z2),
                # Back side face (y-axis, back)
                ("y", y2, x1, x2, z1, z2),
            ]

            # Check each side
            for side_axis, side_pos, start1, end1, start2, end2 in sides:
                # Skip sides that are at bin boundaries
                if (side_axis == "x" and (side_pos == 0 or side_pos == self.bin_dims.width)) or (
                    side_axis == "y" and (side_pos == 0 or side_pos == self.bin_dims.length)
                ):
                    continue

                # Calculate side area
                if side_axis == "x":
                    side_area = (end1 - start1) * (end2 - start2)  # length * height
                else:  # y-axis
                    side_area = (end1 - start1) * (end2 - start2)  # width * height

                total_sides += 1

                # Check if this side has support from other items
                supported_area = 0

                for j, other_item in items_df.iterrows():
                    if i == j:  # Skip self
                        continue

                    # Calculate other item coordinates
                    ox1, oy1, oz1 = other_item["x"], other_item["y"], other_item["z"]
                    ox2, oy2, oz2 = (
                        ox1 + other_item["width"],
                        oy1 + other_item["length"],
                        oz1 + other_item["height"],
                    )

                    # Check if other item is adjacent to this side
                    if side_axis == "x":
                        if side_pos == x1:  # Left side
                            if (
                                abs(ox2 - x1) < 0.1
                            ):  # Other item's right side touches this item's left side
                                # Calculate overlap area
                                y_overlap = max(0, min(y2, oy2) - max(y1, oy1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += y_overlap * z_overlap
                        else:  # Right side
                            if (
                                abs(ox1 - x2) < 0.1
                            ):  # Other item's left side touches this item's right side
                                # Calculate overlap area
                                y_overlap = max(0, min(y2, oy2) - max(y1, oy1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += y_overlap * z_overlap
                    else:  # y-axis
                        if side_pos == y1:  # Front side
                            if (
                                abs(oy2 - y1) < 0.1
                            ):  # Other item's back side touches this item's front side
                                # Calculate overlap area
                                x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += x_overlap * z_overlap
                        else:  # Back side
                            if (
                                abs(oy1 - y2) < 0.1
                            ):  # Other item's front side touches this item's back side
                                # Calculate overlap area
                                x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += x_overlap * z_overlap

                # Calculate support ratio for this side
                support_ratio = supported_area / side_area if side_area > 0 else 0

                # Count side as supported if enough of it is covered
                if support_ratio >= self.min_overlap_ratio:
                    supported_sides += 1

        # Calculate overall side support score
        if total_sides == 0:
            return 0.0

        return supported_sides / total_sides

    def get_item_scores(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Return side support scores for individual items."""
        if items_df.empty:
            return pd.DataFrame(columns=["item", "side_support_score"])

        results = []

        for i, item in items_df.iterrows():
            # Calculate item coordinates for all 6 faces
            x1, y1, z1 = item["x"], item["y"], item["z"]
            x2, y2, z2 = x1 + item["width"], y1 + item["length"], z1 + item["height"]

            # Define the 4 sides of the item
            sides = [
                # Left side face (x-axis, left)
                ("x", x1, y1, y2, z1, z2),
                # Right side face (x-axis, right)
                ("x", x2, y1, y2, z1, z2),
                # Front side face (y-axis, front)
                ("y", y1, x1, x2, z1, z2),
                # Back side face (y-axis, back)
                ("y", y2, x1, x2, z1, z2),
            ]

            # Check each side
            total_sides = 0
            supported_sides = 0

            for side_axis, side_pos, start1, end1, start2, end2 in sides:
                # Skip sides that are at bin boundaries
                if (side_axis == "x" and (side_pos == 0 or side_pos == self.bin_dims.width)) or (
                    side_axis == "y" and (side_pos == 0 or side_pos == self.bin_dims.length)
                ):
                    continue

                # Calculate side area
                if side_axis == "x":
                    side_area = (end1 - start1) * (end2 - start2)  # length * height
                else:  # y-axis
                    side_area = (end1 - start1) * (end2 - start2)  # width * height

                total_sides += 1

                # Check if this side has support from other items
                supported_area = 0

                for j, other_item in items_df.iterrows():
                    if i == j:  # Skip self
                        continue

                    # Calculate other item coordinates
                    ox1, oy1, oz1 = other_item["x"], other_item["y"], other_item["z"]
                    ox2, oy2, oz2 = (
                        ox1 + other_item["width"],
                        oy1 + other_item["length"],
                        oz1 + other_item["height"],
                    )

                    # Check if other item is adjacent to this side
                    if side_axis == "x":
                        if side_pos == x1:  # Left side
                            if (
                                abs(ox2 - x1) < 0.1
                            ):  # Other item's right side touches this item's left side
                                # Calculate overlap area
                                y_overlap = max(0, min(y2, oy2) - max(y1, oy1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += y_overlap * z_overlap
                        else:  # Right side
                            if (
                                abs(ox1 - x2) < 0.1
                            ):  # Other item's left side touches this item's right side
                                # Calculate overlap area
                                y_overlap = max(0, min(y2, oy2) - max(y1, oy1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += y_overlap * z_overlap
                    else:  # y-axis
                        if side_pos == y1:  # Front side
                            if (
                                abs(oy2 - y1) < 0.1
                            ):  # Other item's back side touches this item's front side
                                # Calculate overlap area
                                x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += x_overlap * z_overlap
                        else:  # Back side
                            if (
                                abs(oy1 - y2) < 0.1
                            ):  # Other item's front side touches this item's back side
                                # Calculate overlap area
                                x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                                z_overlap = max(0, min(z2, oz2) - max(z1, oz1))
                                supported_area += x_overlap * z_overlap

                # Calculate support ratio for this side
                support_ratio = supported_area / side_area if side_area > 0 else 0

                # Count side as supported if enough of it is covered
                if support_ratio >= self.min_overlap_ratio:
                    supported_sides += 1

            # Calculate side support score for this item
            item_score = supported_sides / total_sides if total_sides > 0 else 0

            results.append({"item": item["item"], "side_support_score": item_score})

        return pd.DataFrame(results)


class SurfaceSupport:
    """Class for calculating the Surface Support KPI."""

    def __init__(
        self,
        bin_dims: BinDimensions,
        min_surface_ratio: float = 0.5,
        corner_support_threshold: int = 3,
    ):
        """Initialize with bin dimensions and support thresholds."""
        self.bin_dims = bin_dims
        self.min_surface_ratio = min_surface_ratio
        self.corner_support_threshold = corner_support_threshold

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the Surface Support KPI."""
        if items_df.empty:
            return 0.0

        # Copy the dataframe to avoid modifying the original
        items_df = items_df.copy()

        # Sort by z-coordinate to process items from bottom to top
        items_df = items_df.sort_values("z")

        # Track support scores for each item
        support_scores = []

        for i, item in items_df.iterrows():
            # Skip items directly on the ground (they have perfect support)
            if abs(item["z"]) < 0.1:
                support_scores.append(1.0)
                continue

            # Calculate item bottom surface coordinates
            x1, y1, z1 = item["x"], item["y"], item["z"]
            x2, y2 = x1 + item["width"], y1 + item["length"]
            bottom_area = item["width"] * item["length"]

            # Calculate the corners of the bottom face
            corners = [
                (x1, y1),  # bottom-left
                (x2, y1),  # bottom-right
                (x1, y2),  # top-left
                (x2, y2),  # top-right
            ]

            supported_corners = 0
            supported_area = 0

            # Check support from other items
            for _, other_item in items_df.iterrows():
                # Skip self or items above this one
                if other_item["item"] == item["item"] or other_item["z"] >= item["z"]:
                    continue

                # Calculate other item top surface coordinates
                ox1, oy1 = other_item["x"], other_item["y"]
                ox2, oy2 = ox1 + other_item["width"], oy1 + other_item["length"]
                oz2 = other_item["z"] + other_item["height"]

                # Check if other item's top is directly below this item's bottom
                if abs(oz2 - z1) < 0.1:
                    # Calculate overlap area
                    x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                    y_overlap = max(0, min(y2, oy2) - max(y1, oy1))

                    if x_overlap > 0 and y_overlap > 0:
                        supported_area += x_overlap * y_overlap

                    # Check if corners are supported
                    for cx, cy in corners:
                        if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                            supported_corners += 1

            # Calculate support ratio
            area_ratio = supported_area / bottom_area if bottom_area > 0 else 0

            # Determine if item has sufficient support
            has_area_support = area_ratio >= self.min_surface_ratio
            has_corner_support = supported_corners >= self.corner_support_threshold

            if has_area_support or has_corner_support:
                # Calculate weighted score based on support type
                if has_area_support and has_corner_support:
                    score = 1.0  # Both types of support: perfect
                elif has_area_support:
                    score = area_ratio  # Area support: score based on coverage
                else:
                    score = supported_corners / 4  # Corner support: score based on corner count
            else:
                # Insufficient support
                score = area_ratio  # Partial score based on whatever support exists

            support_scores.append(score)

        # Calculate overall score as average of all item scores
        if not support_scores:
            return 0.0

        return sum(support_scores) / len(support_scores)

    def get_item_scores(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Return surface support scores for individual items."""
        if items_df.empty:
            return pd.DataFrame(
                columns=["item", "area_support_ratio", "corner_support_count", "support_score"]
            )

        # Copy and sort the dataframe
        items_df = items_df.copy().sort_values("z")

        results = []

        for i, item in items_df.iterrows():
            # Calculate item bottom surface coordinates
            x1, y1, z1 = item["x"], item["y"], item["z"]
            x2, y2 = x1 + item["width"], y1 + item["length"]
            bottom_area = item["width"] * item["length"]

            # Handle items directly on the ground
            if abs(z1) < 0.1:
                results.append(
                    {
                        "item": item["item"],
                        "area_support_ratio": 1.0,
                        "corner_support_count": 4,
                        "support_score": 1.0,
                    }
                )
                continue

            # Calculate the corners of the bottom face
            corners = [
                (x1, y1),  # bottom-left
                (x2, y1),  # bottom-right
                (x1, y2),  # top-left
                (x2, y2),  # top-right
            ]

            supported_corners = 0
            supported_area = 0

            # Check support from other items
            for _, other_item in items_df.iterrows():
                # Skip self or items above this one
                if other_item["item"] == item["item"] or other_item["z"] >= item["z"]:
                    continue

                # Calculate other item top surface coordinates
                ox1, oy1 = other_item["x"], other_item["y"]
                ox2, oy2 = ox1 + other_item["width"], oy1 + other_item["length"]
                oz2 = other_item["z"] + other_item["height"]

                # Check if other item's top is directly below this item's bottom
                if abs(oz2 - z1) < 0.1:
                    # Calculate overlap area
                    x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                    y_overlap = max(0, min(y2, oy2) - max(y1, oy1))

                    if x_overlap > 0 and y_overlap > 0:
                        supported_area += x_overlap * y_overlap

                    # Check if corners are supported
                    for idx, (cx, cy) in enumerate(corners):
                        if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                            supported_corners += 1

            # Calculate area support ratio
            area_ratio = supported_area / bottom_area if bottom_area > 0 else 0

            # Determine if item has sufficient support
            has_area_support = area_ratio >= self.min_surface_ratio
            has_corner_support = supported_corners >= self.corner_support_threshold

            if has_area_support or has_corner_support:
                # Calculate weighted score based on support type
                if has_area_support and has_corner_support:
                    score = 1.0  # Both types of support: perfect
                elif has_area_support:
                    score = area_ratio  # Area support: score based on coverage
                else:
                    score = supported_corners / 4  # Corner support: score based on corner count
            else:
                # Insufficient support
                score = area_ratio  # Partial score based on whatever support exists

            results.append(
                {
                    "item": item["item"],
                    "area_support_ratio": area_ratio,
                    "corner_support_count": supported_corners,
                    "support_score": score,
                }
            )

        return pd.DataFrame(results)


class CenterOfGravity2D:
    """Class for calculating the Center of Gravity 2D KPI."""

    def __init__(self, bin_dims: BinDimensions):
        """Initialize with bin dimensions."""
        self.bin_dims = bin_dims

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the Center of Gravity 2D KPI."""
        if items_df.empty:
            return 0.0

        # Calculate center of each item and its mass (volume * density)
        # Assuming uniform density for all items
        items_df = items_df.copy()
        items_df["mass"] = items_df["width"] * items_df["length"] * items_df["height"]
        items_df["center_x"] = items_df["x"] + items_df["width"] / 2
        items_df["center_y"] = items_df["y"] + items_df["length"] / 2

        # Calculate center of gravity
        total_mass = items_df["mass"].sum()

        if total_mass == 0:
            return 0.0

        cog_x = np.sum(items_df["center_x"] * items_df["mass"]) / total_mass
        cog_y = np.sum(items_df["center_y"] * items_df["mass"]) / total_mass

        # Calculate ideal center of gravity (center of bin)
        ideal_x = self.bin_dims.width / 2
        ideal_y = self.bin_dims.length / 2

        # Calculate worst possible distance (from center to corner)
        worst_distance = np.sqrt(ideal_x**2 + ideal_y**2)

        # Calculate actual distance from ideal
        actual_distance = np.sqrt((cog_x - ideal_x) ** 2 + (cog_y - ideal_y) ** 2)

        # Normalize to 0-1 score (1 = perfect, 0 = worst)
        if worst_distance == 0:
            return 1.0

        score = 1.0 - (actual_distance / worst_distance)

        return max(0.0, min(1.0, score))

    def get_cog_coordinates(self, items_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate and return the center of gravity coordinates."""
        if items_df.empty:
            return {"cog_x": 0, "cog_y": 0}

        # Calculate center of each item and its mass
        items_df = items_df.copy()
        items_df["mass"] = items_df["width"] * items_df["length"] * items_df["height"]
        items_df["center_x"] = items_df["x"] + items_df["width"] / 2
        items_df["center_y"] = items_df["y"] + items_df["length"] / 2

        # Calculate center of gravity
        total_mass = items_df["mass"].sum()

        if total_mass == 0:
            return {"cog_x": 0, "cog_y": 0}

        cog_x = np.sum(items_df["center_x"] * items_df["mass"]) / total_mass
        cog_y = np.sum(items_df["center_y"] * items_df["mass"]) / total_mass

        return {"cog_x": cog_x, "cog_y": cog_y}

    def visualize(self, items_df: pd.DataFrame) -> plt.Figure:
        """Visualize the center of gravity in 2D."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate COG coordinates
        cog = self.get_cog_coordinates(items_df)

        # Plot bin boundaries
        ax.plot(
            [0, 0, self.bin_dims.width, self.bin_dims.width, 0],
            [0, self.bin_dims.length, self.bin_dims.length, 0, 0],
            "k-",
            linewidth=2,
        )

        # Plot items (top view)
        for _, item in items_df.iterrows():
            rect = plt.Rectangle(
                (item["x"], item["y"]),
                item["width"],
                item["length"],
                facecolor="lightblue",
                alpha=0.5,
                edgecolor="blue",
            )
            ax.add_patch(rect)

        # Plot ideal center
        ideal_x = self.bin_dims.width / 2
        ideal_y = self.bin_dims.length / 2
        ax.plot(ideal_x, ideal_y, "go", markersize=10, label="Ideal CoG")

        # Plot actual center of gravity
        ax.plot(cog["cog_x"], cog["cog_y"], "ro", markersize=10, label="Actual CoG")

        # Show distance line
        ax.plot([ideal_x, cog["cog_x"]], [ideal_y, cog["cog_y"]], "r--")

        ax.set_xlim(-0.1 * self.bin_dims.width, 1.1 * self.bin_dims.width)
        ax.set_ylim(-0.1 * self.bin_dims.length, 1.1 * self.bin_dims.length)
        ax.set_aspect("equal")
        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_title("2D Center of Gravity Analysis")
        ax.legend()

        return fig


class CenterOfGravity3D:
    """Class for calculating the Center of Gravity 3D KPI."""

    def __init__(self, bin_dims: BinDimensions):
        """Initialize with bin dimensions."""
        self.bin_dims = bin_dims

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the Center of Gravity 3D KPI."""
        if items_df.empty:
            return 0.0

        # Calculate center of each item and its mass
        items_df = items_df.copy()
        items_df["mass"] = items_df["width"] * items_df["length"] * items_df["height"]
        items_df["center_x"] = items_df["x"] + items_df["width"] / 2
        items_df["center_y"] = items_df["y"] + items_df["length"] / 2
        items_df["center_z"] = items_df["z"] + items_df["height"] / 2

        # Calculate center of gravity
        total_mass = items_df["mass"].sum()

        if total_mass == 0:
            return 0.0

        cog_x = np.sum(items_df["center_x"] * items_df["mass"]) / total_mass
        cog_y = np.sum(items_df["center_y"] * items_df["mass"]) / total_mass
        cog_z = np.sum(items_df["center_z"] * items_df["mass"]) / total_mass

        # Calculate ideal center of gravity (center of bin)
        ideal_x = self.bin_dims.width / 2
        ideal_y = self.bin_dims.length / 2
        ideal_z = self.bin_dims.height / 2

        # Calculate worst possible distance (from center to corner)
        worst_distance = np.sqrt(ideal_x**2 + ideal_y**2 + ideal_z**2)

        # Calculate actual distance from ideal
        actual_distance = np.sqrt(
            (cog_x - ideal_x) ** 2 + (cog_y - ideal_y) ** 2 + (cog_z - ideal_z) ** 2
        )

        # Normalize to 0-1 score (1 = perfect, 0 = worst)
        if worst_distance == 0:
            return 1.0

        score = 1.0 - (actual_distance / worst_distance)

        return max(0.0, min(1.0, score))

    def get_cog_coordinates(self, items_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate and return the center of gravity coordinates in 3D."""
        if items_df.empty:
            return {"cog_x": 0, "cog_y": 0, "cog_z": 0}

        # Calculate center of each item and its mass
        items_df = items_df.copy()
        items_df["mass"] = items_df["width"] * items_df["length"] * items_df["height"]
        items_df["center_x"] = items_df["x"] + items_df["width"] / 2
        items_df["center_y"] = items_df["y"] + items_df["length"] / 2
        items_df["center_z"] = items_df["z"] + items_df["height"] / 2

        # Calculate center of gravity
        total_mass = items_df["mass"].sum()

        if total_mass == 0:
            return {"cog_x": 0, "cog_y": 0, "cog_z": 0}

        cog_x = np.sum(items_df["center_x"] * items_df["mass"]) / total_mass
        cog_y = np.sum(items_df["center_y"] * items_df["mass"]) / total_mass
        cog_z = np.sum(items_df["center_z"] * items_df["mass"]) / total_mass

        return {"cog_x": cog_x, "cog_y": cog_y, "cog_z": cog_z}

    def visualize(self, items_df: pd.DataFrame) -> plt.Figure:
        """Visualize the center of gravity in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Calculate COG coordinates
        cog = self.get_cog_coordinates(items_df)

        # Plot bin boundaries
        # Bottom face
        x = [0, self.bin_dims.width, self.bin_dims.width, 0, 0]
        y = [0, 0, self.bin_dims.length, self.bin_dims.length, 0]
        z = [0, 0, 0, 0, 0]
        ax.plot3D(x, y, z, "k-")

        # Top face
        x = [0, self.bin_dims.width, self.bin_dims.width, 0, 0]
        y = [0, 0, self.bin_dims.length, self.bin_dims.length, 0]
        z = [
            self.bin_dims.height,
            self.bin_dims.height,
            self.bin_dims.height,
            self.bin_dims.height,
            self.bin_dims.height,
        ]
        ax.plot3D(x, y, z, "k-")

        # Vertical edges
        for x, y in [
            (0, 0),
            (self.bin_dims.width, 0),
            (self.bin_dims.width, self.bin_dims.length),
            (0, self.bin_dims.length),
        ]:
            ax.plot3D([x, x], [y, y], [0, self.bin_dims.height], "k-")

        # Plot items as wireframes
        for _, item in items_df.iterrows():
            x1, y1, z1 = item["x"], item["y"], item["z"]
            x2, y2, z2 = x1 + item["width"], y1 + item["length"], z1 + item["height"]

            # Bottom face
            x = [x1, x2, x2, x1, x1]
            y = [y1, y1, y2, y2, y1]
            z = [z1, z1, z1, z1, z1]
            ax.plot3D(x, y, z, "b-", alpha=0.3)

            # Top face
            x = [x1, x2, x2, x1, x1]
            y = [y1, y1, y2, y2, y1]
            z = [z2, z2, z2, z2, z2]
            ax.plot3D(x, y, z, "b-", alpha=0.3)

            # Vertical edges
            for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                ax.plot3D([x, x], [y, y], [z1, z2], "b-", alpha=0.3)

        # Plot ideal center
        ideal_x = self.bin_dims.width / 2
        ideal_y = self.bin_dims.length / 2
        ideal_z = self.bin_dims.height / 2
        ax.scatter([ideal_x], [ideal_y], [ideal_z], color="g", s=100, label="Ideal CoG")

        # Plot actual center of gravity
        ax.scatter(
            [cog["cog_x"]], [cog["cog_y"]], [cog["cog_z"]], color="r", s=100, label="Actual CoG"
        )

        # Show distance line
        ax.plot([ideal_x, cog["cog_x"]], [ideal_y, cog["cog_y"]], [ideal_z, cog["cog_z"]], "r--")

        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_zlabel("Height")
        ax.set_title("3D Center of Gravity Analysis")
        ax.legend()

        return fig


class AbsoluteDensity:
    """Class for calculating the Absolute Density KPI."""

    def __init__(self, bin_dims: BinDimensions):
        """Initialize with bin dimensions."""
        self.bin_dims = bin_dims

    def calculate(self, items_df: pd.DataFrame) -> float:
        """Calculate the Absolute Density KPI."""
        if items_df.empty:
            return 0.0

        # Calculate total volume of all items
        items_df = items_df.copy()
        items_df["volume"] = items_df["width"] * items_df["length"] * items_df["height"]
        total_items_volume = items_df["volume"].sum()

        # Calculate bin volume
        bin_volume = self.bin_dims.volume

        # Calculate absolute density
        if bin_volume == 0:
            return 0.0

        absolute_density = total_items_volume / bin_volume

        # Ensure the score is between 0 and 1
        return max(0.0, min(1.0, absolute_density))

    def get_volume_breakdown(self, items_df: pd.DataFrame) -> Dict[str, float]:
        """Return detailed volume breakdown."""
        if items_df.empty:
            return {
                "bin_volume": self.bin_dims.volume,
                "total_items_volume": 0.0,
                "volume_utilization": 0.0,
            }

        # Calculate volumes
        items_df = items_df.copy()
        items_df["volume"] = items_df["width"] * items_df["length"] * items_df["height"]
        total_items_volume = items_df["volume"].sum()
        bin_volume = self.bin_dims.volume

        # Calculate utilization percentage
        volume_utilization = total_items_volume / bin_volume if bin_volume > 0 else 0.0

        return {
            "bin_volume": bin_volume,
            "total_items_volume": total_items_volume,
            "volume_utilization": volume_utilization,
        }

    def visualize(self, items_df: pd.DataFrame) -> plt.Figure:
        """Visualize the volume utilization."""
        # Get volume breakdown
        volume_data = self.get_volume_breakdown(items_df)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot bin volume as 100%
        bin_volume = volume_data["bin_volume"]
        items_volume = volume_data["total_items_volume"]
        empty_volume = max(0, bin_volume - items_volume)

        # Create stacked bar
        ax.bar(["Bin Volume"], [items_volume], label="Items Volume", color="green")
        ax.bar(
            ["Bin Volume"],
            [empty_volume],
            bottom=[items_volume],
            label="Empty Volume",
            color="lightgray",
        )

        # Add utilization percentage
        utilization_pct = volume_data["volume_utilization"] * 100
        ax.text(
            0,
            bin_volume / 2,
            f"{utilization_pct:.1f}%",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        # Add volume values
        ax.text(1.05, items_volume / 2, f"{items_volume:.0f} cubic units", va="center", ha="left")
        ax.text(
            1.05,
            items_volume + empty_volume / 2,
            f"{empty_volume:.0f} cubic units",
            va="center",
            ha="left",
        )

        ax.set_ylim(0, bin_volume * 1.1)
        ax.set_ylabel("Volume")
        ax.set_title("Bin Volume Utilization")
        ax.legend(loc="upper right")

        return fig


class BedBppKPIEvaluator:
    """BED-BPP Benchmark: KPI Evaluator (Updated with BinPackingEvaluator KPIs)."""

    def __init__(self, bin_dims: BinDimensions, weights: Dict[str, float] = None):
        """Args:."""
        self.bin_dims = bin_dims

        # Set default weights if not provided
        if weights is None:
            self.weights = {
                "absolute_density": 1.0,
                "height_width_ratio": 1.0,
                "relative_density": 0.5,
                "side_support": 0.5,
                "surface_support": 0.5,
                "center_of_gravity_2d": 0.2,
                "center_of_gravity_3d": 0.2,
            }
        else:
            self.weights = weights

        # Initialize KPI calculators from BinPackingEvaluator
        # Import classes from the attached file (they're already in the same file)

        self.height_width_ratio = HeightWidthRatio(bin_dims)
        self.relative_density = RelativeDensity(bin_dims)
        self.absolute_density = AbsoluteDensity(bin_dims)
        self.side_support = SideSupport(bin_dims)
        self.surface_support = SurfaceSupport(bin_dims)
        self.center_of_gravity_2d = CenterOfGravity2D(bin_dims)
        self.center_of_gravity_3d = CenterOfGravity3D(bin_dims)

    def evaluate(self, items_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate all KPIs using BinPackingEvaluator methodology."""
        # Calculate individual KPI scores
        height_width_ratio_score = self.height_width_ratio.calculate(items_df)
        absolute_density_score = self.absolute_density.calculate(items_df)
        relative_density_score = self.relative_density.calculate(items_df)
        side_support_score = self.side_support.calculate(items_df)
        surface_support_score = self.surface_support.calculate(items_df)
        cog_2d_score = self.center_of_gravity_2d.calculate(items_df)
        cog_3d_score = self.center_of_gravity_3d.calculate(items_df)

        # Calculate overall score (weighted average)
        total_weight = sum(self.weights.values())

        if total_weight == 0:
            overall_score = 0.0
        else:
            overall_score = (
                self.weights["absolute_density"] * absolute_density_score
                + self.weights["height_width_ratio"] * height_width_ratio_score
                + self.weights["relative_density"] * relative_density_score
                + self.weights["side_support"] * side_support_score
                + self.weights["surface_support"] * surface_support_score
                + self.weights["center_of_gravity_2d"] * cog_2d_score
                + self.weights["center_of_gravity_3d"] * cog_3d_score
            ) / total_weight

        # Construct result dictionary
        result = {
            "absolute_density": absolute_density_score,
            "relative_density": relative_density_score,
            "side_support": side_support_score,
            "surface_support": surface_support_score,
            "center_of_gravity_2d": cog_2d_score,
            "center_of_gravity_3d": cog_3d_score,
            "height_width_ratio": height_width_ratio_score,
            "overall_score": overall_score,
        }

        return result

    def get_kpi_vector(self, kpis: dict) -> list:
        """Returns the 7D KPI vector from BinPackingEvaluator."""
        return [
            kpis["absolute_density"],
            kpis["relative_density"],
            kpis["side_support"],
            kpis["surface_support"],
            kpis["center_of_gravity_2d"],
            kpis["center_of_gravity_3d"],
            kpis["height_width_ratio"],
        ]

    def evaluate_detailed(self, items_df: pd.DataFrame) -> Dict:
        """Perform detailed evaluation with item-level metrics."""
        # Overall scores
        overall_scores = self.evaluate(items_df)

        # Item-level scores
        height_width_ratio_details = self.height_width_ratio.get_item_scores(items_df)
        side_support_details = self.side_support.get_item_scores(items_df)
        surface_support_details = self.surface_support.get_item_scores(items_df)

        # Volume details for absolute density
        volume_details = self.absolute_density.get_volume_breakdown(items_df)

        # Relative density details
        relative_density_details = self.relative_density.calculate_with_details(items_df)

        # Center of gravity details
        cog_2d = self.center_of_gravity_2d.get_cog_coordinates(items_df)
        cog_3d = self.center_of_gravity_3d.get_cog_coordinates(items_df)

        # Compile detailed results
        detailed_results = {
            "overall_scores": overall_scores,
            "item_details": {
                "height_width_ratio": height_width_ratio_details.to_dict("records"),
                "side_support": side_support_details.to_dict("records"),
                "surface_support": surface_support_details.to_dict("records"),
            },
            "volume_details": volume_details,
            "relative_density_details": relative_density_details,
            "center_of_gravity": {"2d": cog_2d, "3d": cog_3d},
        }

        return detailed_results

    def generate_report(self, items_df: pd.DataFrame) -> str:
        """Generate a text report summarizing the evaluation."""
        # Evaluate all KPIs
        scores = self.evaluate(items_df)

        # Get volume details for the report
        volume_details = self.absolute_density.get_volume_breakdown(items_df)
        relative_density_details = self.relative_density.calculate_with_details(items_df)

        # Format report
        report = "BIN PACKING EVALUATION REPORT\n"
        report += "============================\n\n"

        report += f"Bin dimensions: {self.bin_dims.width} x {self.bin_dims.length} x {self.bin_dims.height}\n"
        report += f"Bin volume: {self.bin_dims.volume:.2f} cubic units\n"
        report += f"Total items volume: {volume_details['total_items_volume']:.2f} cubic units\n"
        report += f"Volume utilization: {volume_details['volume_utilization']*100:.2f}%\n"
        report += f"Number of items: {len(items_df)}\n\n"

        report += "DENSITY ANALYSIS:\n"
        report += "----------------\n"
        report += (
            f"Absolute Density: {scores['absolute_density']:.4f} (items volume / bin volume)\n"
        )
        report += (
            f"Relative Density: {scores['relative_density']:.4f} (items volume / utilized space)\n"
        )
        report += (
            f"Utilized Volume: {relative_density_details['utilized_volume']:.2f} cubic units\n"
        )
        report += (
            f"Waste in Utilized Space: {relative_density_details['waste_percentage']:.2f}%\n\n"
        )

        report += "KPI SCORES (0-1 scale, 1 = optimal):\n"
        report += "-----------------------------------\n"
        report += (
            f"1. Absolute Density:     {scores['absolute_density']:.4f}  (Volume utilization)\n"
        )
        report += f"2. Relative Density:     {scores['relative_density']:.4f}  (Space utilization efficiency)\n"
        report += (
            f"3. Side Support:         {scores['side_support']:.4f}  (Side adjacency support)\n"
        )
        report += f"4. Surface Support:      {scores['surface_support']:.4f}  (Bottom surface stability)\n"
        report += (
            f"5. Center of Gravity 2D: {scores['center_of_gravity_2d']:.4f}  (Horizontal balance)\n"
        )
        report += (
            f"6. Center of Gravity 3D: {scores['center_of_gravity_3d']:.4f}  (Overall balance)\n"
        )
        report += f"7. Height-Width Ratio:   {scores['height_width_ratio']:.4f}  (Tall item stability)\n\n"
        report += f"OVERALL SCORE:           {scores['overall_score']:.4f}\n\n"

        return report


# --------------- Example Usage -----------------
if __name__ == "__main__":
    # Example: bin is 1200x800x1000
    bin_dims = BinDimensions(1200, 800, 1000)

    # Sample DataFrame with packed items
    df = pd.DataFrame(
        [
            {"item": 1, "x": 0, "y": 0, "z": 0, "width": 400, "length": 300, "height": 200},
            {"item": 2, "x": 400, "y": 0, "z": 0, "width": 300, "length": 400, "height": 150},
            {"item": 3, "x": 0, "y": 300, "z": 0, "width": 250, "length": 250, "height": 300},
            {"item": 4, "x": 700, "y": 0, "z": 0, "width": 200, "length": 300, "height": 180},
        ]
    )

    # Create evaluator with custom weights
    weights = {
        "absolute_density": 2.0,  # Highest priority
        "height_width_ratio": 1.0,
        "relative_density": 1.0,
        "side_support": 1.0,
        "surface_support": 1.0,
        "center_of_gravity_2d": 0.5,
        "center_of_gravity_3d": 0.5,
    }

    evaluator = BedBppKPIEvaluator(bin_dims, weights)

    # Evaluate KPIs
    kpi_results = evaluator.evaluate(df)
    for k, v in kpi_results.items():
        print(f"{k}: {v:.4f}")

    # Generate report
    report = evaluator.generate_report(df)
    print("\n" + report)

    # Get detailed evaluation
    detailed_results = evaluator.evaluate_detailed(df)
    print("Detailed evaluation completed with", len(detailed_results), "sections")

    # To get the KPI vector:
    kpi_vector = evaluator.get_kpi_vector(kpi_results)
    print("KPI vector:", [f"{x:.4f}" for x in kpi_vector])
