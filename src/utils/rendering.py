from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from src.utils import utils, visualization


@dataclass
class RenderConfig:
    use_vtk: bool = False
    vtk_interactive: bool = False
    vtk_resolution: str = "1200x900"


def parse_resolution(resolution):
    """Parse a WIDTHxHEIGHT resolution string."""
    try:
        width_str, height_str = resolution.lower().split("x", maxsplit=1)
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError
        return width, height
    except ValueError as exc:
        raise ValueError(
            f"Invalid --vtk-resolution value '{resolution}'. Expected format WIDTHxHEIGHT, e.g. 1600x1200."
        ) from exc


def create_vtk_visualizer(render_config, pallet_dims):
    """Create a VTK visualizer from rendering configuration."""
    try:
        from src.utils.vtk_visualization import VTKVisualizer

        width, height = parse_resolution(render_config.vtk_resolution)
        return VTKVisualizer(pallet_dims), width, height
    except Exception as e:
        logger.warning(f"Failed to create VTK visualizer: {e}")
        logger.warning("VTK visualization not available")
        return None, None, None


def visualize_stage_vtk(vtk_viz, data, title, filename_base, render_config, width, height):
    """Visualize a specific stage with VTK."""
    if data is None or len(data) == 0:
        return

    if render_config.vtk_interactive:
        vtk_viz.visualize_packing(data, title=title, width=width, height=height, interactive=True)
    else:
        vtk_viz.visualize_packing(
            data, title=title, filename=f"results/{filename_base}", width=width, height=height
        )


def save_dataframe_plot(data, pallet_dims, output_path, title):
    """Render a packing DataFrame to a static matplotlib image."""
    if data is None or len(data) == 0:
        return

    ax = visualization.get_pallet_plot(pallet_dims)
    ax.set_title(title)
    for row in data.itertuples(index=False):
        coords = utils.Coordinate(row.x, row.y, row.z)
        dims = utils.Dimension(row.width, row.length, row.height, getattr(row, "weight", 0))
        visualization.plot_product(ax, row.item, coords, dims, pallet_dims)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)


def visualize_bins_solution(
    bin_pool,
    render_config,
    pallet_dims,
    phase_name="",
    output_file_base="",
    vtk_viz=None,
    width=None,
    height=None,
):
    """Visualize bin packing solutions using VTK or matplotlib."""
    if render_config.use_vtk:
        if vtk_viz is None:
            vtk_viz, width, height = create_vtk_visualizer(render_config, pallet_dims)

        if vtk_viz is not None:
            try:
                results_dir = Path("results")
                results_dir.mkdir(parents=True, exist_ok=True)

                if hasattr(bin_pool, "compact_bins"):
                    bin_dataframes = []
                    for i, compact_bin in enumerate(bin_pool.compact_bins):
                        df_bin = compact_bin.to_dataframe()
                        df_bin["bin"] = i
                        bin_dataframes.append(df_bin)

                    for i, df_bin in enumerate(bin_dataframes):
                        filename = results_dir / f"vtk_bin_{i+1}_{phase_name}_{output_file_base}.png"
                        vtk_viz.visualize_packing(
                            df_bin,
                            title=f"Bin {i+1} - {phase_name}",
                            filename=str(filename),
                            width=width,
                            height=height,
                        )
                elif hasattr(bin_pool, "to_dataframe"):
                    df_bin = bin_pool.to_dataframe()
                    filename = results_dir / f"vtk_{phase_name}_{output_file_base}.png"
                    vtk_viz.visualize_packing(
                        df_bin,
                        title=f"{phase_name} Solution",
                        filename=str(filename),
                        width=width,
                        height=height,
                    )
                else:
                    if render_config.vtk_interactive:
                        vtk_viz.visualize_packing(
                            bin_pool,
                            title=f"{phase_name} Solution",
                            width=width,
                            height=height,
                            interactive=True,
                        )
                    else:
                        filename = results_dir / f"vtk_{phase_name}_{output_file_base}.png"
                        vtk_viz.visualize_packing(
                            bin_pool,
                            title=f"{phase_name} Solution",
                            filename=str(filename),
                            width=width,
                            height=height,
                        )

                logger.debug(f"VTK visualization saved for {phase_name}")
                return
            except Exception as e:
                logger.error(f"VTK visualization failed: {e}")

    output_path = Path("results") / f"{phase_name}_{output_file_base}.png"
    if hasattr(bin_pool, "to_dataframe"):
        save_dataframe_plot(
            bin_pool.to_dataframe(), pallet_dims, output_path, f"{phase_name} Solution"
        )
    else:
        save_dataframe_plot(bin_pool, pallet_dims, output_path, f"{phase_name} Solution")
    logger.debug(f"Matplotlib visualization saved for {phase_name}")
