"""VTK-Based 3D Visualization for Bin Packing."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import vtk
from loguru import logger

# VTK Color palette (from original implementation)
VTK_COLOR_PALETTE = {
    "Whites": [
        "antique_white",
        "azure",
        "bisque",
        "blanched_almond",
        "cornsilk",
        "eggshell",
        "floral_white",
        "gainsboro",
        "ghost_white",
        "honeydew",
        "ivory",
        "lavender",
        "lavender_blush",
        "lemon_chiffon",
        "linen",
        "mint_cream",
        "misty_rose",
        "moccasin",
        "navajo_white",
        "old_lace",
        "papaya_whip",
        "peach_puff",
        "seashell",
        "snow",
        "thistle",
        "titanium_white",
        "wheat",
        "white",
        "white_smoke",
        "zinc_white",
    ],
    "Greys": [
        "cold_grey",
        "dim_grey",
        "grey",
        "light_grey",
        "slate_grey",
        "slate_grey_dark",
        "slate_grey_light",
        "warm_grey",
    ],
    "Reds": [
        "coral",
        "coral_light",
        "hot_pink",
        "light_salmon",
        "pink",
        "pink_light",
        "raspberry",
        "rose_madder",
        "salmon",
    ],
    "Oranges": [
        "cadmium_orange",
        "cadmium_red_light",
        "carrot",
        "dark_orange",
        "mars_orange",
        "mars_yellow",
        "orange",
        "orange_red",
        "yellow_ochre",
    ],
    "Yellows": [
        "aureoline_yellow",
        "banana",
        "cadmium_lemon",
        "cadmium_yellow",
        "cadmium_yellow_light",
        "gold",
        "goldenrod",
        "goldenrod_dark",
        "goldenrod_light",
        "goldenrod_pale",
        "light_goldenrod",
        "melon",
        "yellow",
        "yellow_light",
    ],
    "Greens": [
        "chartreuse",
        "chrome_oxide_green",
        "cinnabar_green",
        "cobalt_green",
        "emerald_green",
        "forest_green",
        "green_dark",
        "green_pale",
        "green_yellow",
        "lawn_green",
        "lime_green",
        "mint",
        "olive",
        "olive_drab",
        "olive_green_dark",
        "permanent_green",
        "sap_green",
        "sea_green",
        "sea_green_dark",
        "sea_green_medium",
        "sea_green_light",
        "spring_green",
        "spring_green_medium",
        "terre_verte",
        "viridian_light",
        "yellow_green",
    ],
    "Cyans": [
        "aquamarine",
        "aquamarine_medium",
        "cyan",
        "cyan_white",
        "turquoise",
        "turquoise_dark",
        "turquoise_medium",
        "turquoise_pale",
    ],
    "Blues": [
        "alice_blue",
        "blue_light",
        "blue_medium",
        "cadet",
        "cobalt",
        "cornflower",
        "cerulean",
        "dodger_blue",
        "indigo",
        "manganese_blue",
        "midnight_blue",
        "navy",
        "peacock",
        "powder_blue",
        "royal_blue",
        "slate_blue",
        "slate_blue_dark",
        "slate_blue_light",
        "slate_blue_medium",
        "sky_blue",
        "sky_blue_light",
        "steel_blue",
        "steel_blue_light",
        "turquoise_blue",
        "ultramarine",
    ],
    "Magentas": [
        "blue_violet",
        "magenta",
        "orchid",
        "orchid_dark",
        "orchid_medium",
        "plum",
        "purple",
        "purple_medium",
        "ultramarine_violet",
        "violet",
        "violet_dark",
        "violet_red_medium",
        "violet_red_pale",
    ],
}


class VTKColorManager:
    """Manages VTK color assignment using the original color palette."""

    def __init__(self):
        self.item_idx = 0
        self.color_keys = list(VTK_COLOR_PALETTE.keys())
        self.colors = vtk.vtkNamedColors()

    def get_vtk_color_for_item(self, item_idx):
        """Get VTK color using the same logic as original VTKRender class."""
        color_0 = self.color_keys[item_idx % len(self.color_keys)]
        color_1 = int(item_idx / len(self.color_keys))

        try:
            if color_1 < len(VTK_COLOR_PALETTE[color_0]):
                color_name = VTK_COLOR_PALETTE[color_0][color_1]
                return self.colors.GetColor3d(color_name)
            else:
                # If we run out of colors in the group, cycle back
                color_name = VTK_COLOR_PALETTE[color_0][color_1 % len(VTK_COLOR_PALETTE[color_0])]
                return self.colors.GetColor3d(color_name)
        except:
            # Fallback to red if color doesn't exist in VTK
            logger.warning(f"Color '{color_name}' not found in VTK, using red fallback")
            return self.colors.GetColor3d("red")

    def reset_index(self):
        """Reset the item index counter."""
        self.item_idx = 0


class VTKVisualizer:
    """Advanced VTK-based 3D visualization for bin packing solutions."""

    def __init__(self, pallet_dims, color_manager=None):
        """Initialize the VTK visualizer."""
        self.pallet_dims = pallet_dims
        self.color_manager = color_manager or VTKColorManager()

    def create_render_window(
        self, win_size=(1200, 900), interactive=False, window_title="3D Bin Packing Visualization"
    ):
        """Create VTK render window - either interactive or headless."""
        # Create renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95, 0.95, 0.95)  # Light gray background

        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(win_size[0], win_size[1])
        render_window.SetWindowName(window_title)
        render_window.AddRenderer(renderer)

        interactor = None

        if interactive:
            # Interactive mode - create window and interactor
            render_window.SetOffScreenRendering(0)  # Enable on-screen rendering

            # Create interactor for mouse/keyboard interaction
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)

            # Set up interaction style for better 3D navigation
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
        else:
            # Headless mode
            render_window.SetOffScreenRendering(1)  # Force off-screen

        return render_window, renderer, interactor

    def add_pallet_outline(self, renderer):
        """Add pallet outline to the scene."""
        container = vtk.vtkCubeSource()
        container.SetXLength(self.pallet_dims.width)
        container.SetYLength(self.pallet_dims.length)
        container.SetZLength(self.pallet_dims.height)
        container.SetCenter([0, 0, 0])

        container_mapper = vtk.vtkPolyDataMapper()
        container_mapper.SetInputConnection(container.GetOutputPort())

        container_actor = vtk.vtkActor()
        container_actor.SetMapper(container_mapper)
        container_actor.GetProperty().SetColor(0.2, 0.2, 0.2)
        container_actor.GetProperty().SetRepresentationToWireframe()
        container_actor.GetProperty().SetLineWidth(1.0)
        container_actor.GetProperty().SetOpacity(0.8)

        renderer.AddActor(container_actor)

    def add_items_to_scene(self, renderer, items_df):
        """Add items to the VTK scene with proper coloring and positioning."""
        # Sort items by z-level for proper rendering
        items_df = items_df.sort_values(by="z", ascending=True)
        z_groups = items_df.groupby("z")

        # Reset color index
        self.color_manager.reset_index()

        for z_level, group in z_groups:
            for idx, (_, item) in enumerate(group.iterrows()):
                # Z-offset for same-level items to avoid overlap
                z_offset = idx * 2.0 if len(group) > 1 else 0.0

                # Create cube
                cube = vtk.vtkCubeSource()
                cube.SetXLength(item["width"])
                cube.SetYLength(item["length"])
                cube.SetZLength(item["height"])

                # Position cube (convert to VTK coordinate system)
                cube.SetCenter(
                    [
                        -self.pallet_dims.width / 2 + item["width"] / 2 + item["x"],
                        -self.pallet_dims.length / 2 + item["length"] / 2 + item["y"],
                        -self.pallet_dims.height / 2 + item["height"] / 2 + item["z"] + z_offset,
                    ]
                )

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(cube.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # Use VTK color logic
                rgb = self.color_manager.get_vtk_color_for_item(self.color_manager.item_idx)
                actor.GetProperty().SetColor(*rgb)

                # Material properties - solid appearance
                actor.GetProperty().SetOpacity(1.0)  # Completely opaque
                actor.GetProperty().SetSpecular(0.2)
                actor.GetProperty().SetSpecularPower(8)
                actor.GetProperty().SetAmbient(0.4)
                actor.GetProperty().SetDiffuse(0.8)

                # Enable edges for visibility
                actor.GetProperty().EdgeVisibilityOn()

                # Edges for overlapping items
                if len(group) > 1:
                    actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
                    actor.GetProperty().SetLineWidth(1.0)

                renderer.AddActor(actor)
                self.color_manager.item_idx += 1

    def add_axes_and_grid(self, renderer):
        """Add coordinate axes, labels, and gridlines to the scene."""

        # Create simple, clean axes without the messy default labels
        self.add_clean_axes(renderer)

        # Add grid lines for better spatial reference
        self.add_grid_lines(renderer)

        # Add clean coordinate annotations
        self.add_coordinate_annotations(renderer)

    def add_clean_axes(self, renderer):
        """Add clean, simple coordinate axes."""

        # Create three simple line axes
        axes_data = [
            # X-axis (red)
            {
                "start": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2,
                ],
                "end": [
                    -self.pallet_dims.width / 2 + self.pallet_dims.width * 0.3,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2,
                ],
                "color": (0.8, 0.2, 0.2),
                "label": "Width",
                "label_pos": [
                    -self.pallet_dims.width / 2 + self.pallet_dims.width * 0.35,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2,
                ],
            },
            # Y-axis (green)
            {
                "start": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2,
                ],
                "end": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2 + self.pallet_dims.length * 0.3,
                    -self.pallet_dims.height / 2,
                ],
                "color": (0.2, 0.8, 0.2),
                "label": "Length",
                "label_pos": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2 + self.pallet_dims.length * 0.35,
                    -self.pallet_dims.height / 2,
                ],
            },
            # Z-axis (blue)
            {
                "start": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2,
                ],
                "end": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2 + self.pallet_dims.height * 0.3,
                ],
                "color": (0.2, 0.2, 0.8),
                "label": "Height",
                "label_pos": [
                    -self.pallet_dims.width / 2,
                    -self.pallet_dims.length / 2,
                    -self.pallet_dims.height / 2 + self.pallet_dims.height * 0.35,
                ],
            },
        ]

        for axis in axes_data:
            # Create axis line
            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(*axis["start"])
            line_source.SetPoint2(*axis["end"])

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line_source.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*axis["color"])
            actor.GetProperty().SetLineWidth(3.0)

            renderer.AddActor(actor)

            # Add clean text label
            text_source = vtk.vtkVectorText()
            text_source.SetText(axis["label"])

            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())

            text_actor = vtk.vtkActor()
            text_actor.SetMapper(text_mapper)
            text_actor.SetPosition(*axis["label_pos"])

            # Scale text appropriately
            scale_factor = (
                max(self.pallet_dims.width, self.pallet_dims.length, self.pallet_dims.height) / 50
            )
            text_actor.SetScale(scale_factor, scale_factor, scale_factor)

            # Set text color to match axis
            text_actor.GetProperty().SetColor(*axis["color"])

            renderer.AddActor(text_actor)

    def add_grid_lines(self, renderer):
        """Add grid lines to help with spatial reference."""

        # Grid spacing - adjust based on pallet size
        x_spacing = self.pallet_dims.width // 4
        y_spacing = self.pallet_dims.length // 4
        z_spacing = self.pallet_dims.height // 4

        # X-direction grid lines (parallel to X axis)
        for y in range(0, int(self.pallet_dims.length) + 1, y_spacing):
            for z in range(0, int(self.pallet_dims.height) + 1, z_spacing):
                if y == 0 or z == 0:  # Only draw grid on base and back planes
                    line_source = vtk.vtkLineSource()
                    line_source.SetPoint1(
                        -self.pallet_dims.width / 2,
                        -self.pallet_dims.length / 2 + y,
                        -self.pallet_dims.height / 2 + z,
                    )
                    line_source.SetPoint2(
                        self.pallet_dims.width / 2,
                        -self.pallet_dims.length / 2 + y,
                        -self.pallet_dims.height / 2 + z,
                    )

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(line_source.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(0.7, 0.7, 0.7)
                    actor.GetProperty().SetLineWidth(0.5)
                    actor.GetProperty().SetOpacity(0.3)

                    renderer.AddActor(actor)

        # Y-direction grid lines (parallel to Y axis)
        for x in range(0, int(self.pallet_dims.width) + 1, x_spacing):
            for z in range(0, int(self.pallet_dims.height) + 1, z_spacing):
                if x == 0 or z == 0:  # Only draw grid on base and side planes
                    line_source = vtk.vtkLineSource()
                    line_source.SetPoint1(
                        -self.pallet_dims.width / 2 + x,
                        -self.pallet_dims.length / 2,
                        -self.pallet_dims.height / 2 + z,
                    )
                    line_source.SetPoint2(
                        -self.pallet_dims.width / 2 + x,
                        self.pallet_dims.length / 2,
                        -self.pallet_dims.height / 2 + z,
                    )

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(line_source.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(0.7, 0.7, 0.7)
                    actor.GetProperty().SetLineWidth(0.5)
                    actor.GetProperty().SetOpacity(0.3)

                    renderer.AddActor(actor)

        # Z-direction grid lines (parallel to Z axis) - only at corners
        corner_positions = [
            (-self.pallet_dims.width / 2, -self.pallet_dims.length / 2),
            (self.pallet_dims.width / 2, -self.pallet_dims.length / 2),
            (-self.pallet_dims.width / 2, self.pallet_dims.length / 2),
        ]

        for x, y in corner_positions:
            line_source = vtk.vtkLineSource()
            line_source.SetPoint1(x, y, -self.pallet_dims.height / 2)
            line_source.SetPoint2(x, y, self.pallet_dims.height / 2)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line_source.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.7, 0.7, 0.7)
            actor.GetProperty().SetLineWidth(0.5)
            actor.GetProperty().SetOpacity(0.3)

            renderer.AddActor(actor)

    def add_coordinate_annotations(self, renderer):
        """Add clean coordinate text annotations at key points."""

        # Only show origin annotation - keep it simple
        annotations = [
            (0, 0, 0, "(0,0,0)"),
        ]

        for x, y, z, text in annotations:
            # Create text source
            text_source = vtk.vtkVectorText()
            text_source.SetText(text)

            # Create mapper
            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text_source.GetOutputPort())

            # Create actor
            text_actor = vtk.vtkActor()
            text_actor.SetMapper(text_mapper)

            # Position the text near origin
            text_actor.SetPosition(
                -self.pallet_dims.width / 2 + x + 30,
                -self.pallet_dims.length / 2 + y + 30,
                -self.pallet_dims.height / 2 + z + 30,
            )

            # Scale the text smaller
            scale_factor = (
                max(self.pallet_dims.width, self.pallet_dims.length, self.pallet_dims.height) / 80
            )
            text_actor.SetScale(scale_factor, scale_factor, scale_factor)

            # Set text color - subtle gray
            text_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
            text_actor.GetProperty().SetOpacity(0.7)

            renderer.AddActor(text_actor)

    def setup_camera(self, renderer):
        """Setup optimal camera position and orientation."""
        camera = renderer.GetActiveCamera()
        distance = (
            max(self.pallet_dims.width, self.pallet_dims.length, self.pallet_dims.height) * 2.2
        )
        camera.SetPosition(distance * 0.8, distance * 1.0, distance * 0.7)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCamera()
        camera.Zoom(0.65)

    def render_to_image(self, render_window, filename, width=1200, height=900):
        """Render directly to image - completely headless."""
        # Set final size
        render_window.SetSize(width, height)

        # Force render - no display
        render_window.Render()

        # Create image filter
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.SetInputBufferTypeToRGB()
        window_to_image.ReadFrontBufferOff()
        window_to_image.SetScale(1, 1)
        window_to_image.Update()

        # Save image
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()

        logger.debug(f"VTK render saved: {filename}")

    def visualize_packing(
        self,
        items_df,
        title="3D Packing Visualization",
        filename=None,
        width=1200,
        height=900,
        show_pallet_outline=True,
        show_axes=True,
        show_grid=True,
        interactive=False,
    ):
        """Main visualization method for a single packing solution."""
        # Create render window (interactive or headless) with custom title
        render_window, renderer, interactor = self.create_render_window(
            (width, height), interactive=interactive, window_title=title
        )

        # Add pallet outline if requested
        if show_pallet_outline:
            self.add_pallet_outline(renderer)

        # Add axes and grid if requested
        if show_axes or show_grid:
            self.add_axes_and_grid(renderer)

        # Add items to scene
        self.add_items_to_scene(renderer, items_df)

        # Setup camera
        self.setup_camera(renderer)

        if interactive:
            # Interactive mode - show window
            logger.debug(f"Opening interactive VTK window: {title}")

            render_window.Render()
            interactor.Start()  # This will block until user closes window

            logger.debug("Interactive window closed")
            return None
        else:
            # Headless mode - save to file
            if filename is None:
                safe_title = "".join(
                    c for c in title if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()
                filename = f"vtk_{safe_title.replace(' ', '_')}.png"

            # Ensure output directory exists
            output_dir = Path(filename).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Render to image
            self.render_to_image(render_window, filename, width, height)

            logger.debug(f"Visualization complete: {len(items_df)} items rendered to {filename}")
            return filename

    def visualize_multiple_bins(
        self, bin_dataframes, titles=None, output_dir="vtk_output", width=1200, height=900
    ):
        """Visualize multiple bins and create individual images."""
        if titles is None:
            titles = [f"Bin_{i+1}" for i in range(len(bin_dataframes))]

        output_paths = []

        for i, (df, title) in enumerate(zip(bin_dataframes, titles)):
            filename = Path(output_dir) / f"{title}.png"
            path = self.visualize_packing(df, title, str(filename), width, height)
            output_paths.append(path)

        logger.info(
            f"Multiple bin visualization complete: {len(output_paths)} images saved to {output_dir}"
        )
        return output_paths


def create_vtk_visualizer_from_config(config):
    """Factory function to create VTK visualizer from configuration."""
    return VTKVisualizer(config.PALLET_DIMS)


# Example usage and testing
if __name__ == "__main__":
    # Test data structure
    class TestDimension:
        def __init__(self, width, length, height):
            self.width = width
            self.length = length
            self.height = height

    # Create test data
    test_pallet = TestDimension(800, 1200, 2000)

    # Sample items data
    test_items = pd.DataFrame(
        {
            "x": [0, 100, 200],
            "y": [0, 150, 300],
            "z": [0, 0, 100],
            "width": [100, 150, 120],
            "length": [150, 200, 180],
            "height": [100, 120, 150],
            "item": ["Item_1", "Item_2", "Item_3"],
        }
    )

    # Create visualizer and test
    visualizer = VTKVisualizer(test_pallet)

    try:
        output_file = visualizer.visualize_packing(
            test_items, title="Test Packing", filename="test_vtk_output.png"
        )
        print(f"✅ Test successful! Output saved to: {output_file}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: VTK may not be available in this environment")
