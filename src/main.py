import argparse
import os
import time
from pathlib import Path

from loguru import logger

from src import config
from src.models import bins, ga_optimizer, kpi_analysis, layers, maxrects, superitems
from src.models.dataset import ProductDataset
from src.utils import utils, visualization
from src.utils.rendering import (
    RenderConfig,
    create_vtk_visualizer,
    visualize_bins_solution,
    visualize_stage_vtk,
)


def evaluate_stage_kpi(data, stage_name, pallet_dims):
    """Evaluate KPI for a specific stage."""
    if data is None or len(data) == 0:
        logger.warning(f"No data for {stage_name} KPI analysis")
        return None

    bin_dims = kpi_analysis.BinDimensions(
        width=pallet_dims.width,
        length=pallet_dims.length,
        height=pallet_dims.height,
    )
    evaluator = kpi_analysis.BedBppKPIEvaluator(bin_dims)
    scores = evaluator.evaluate(data)
    report = evaluator.generate_report(data)

    logger.info(f"{stage_name} KPI Report:")
    logger.info(report)

    return scores


def main():
    """Main entry point for GENPACK 3D bin packing solver."""
    parser = argparse.ArgumentParser(description="3D Bin Packing Solver with MAXRECTS")

    # Adding arguments to the parser
    parser.add_argument(
        "--ordered-products-path",
        type=str,
        default="data/1.csv",
        help="Path to the products data file (order CSV).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator."
    )
    parser.add_argument("--max-iters", type=int, default=1, help="Number of iterations to attempt")
    parser.add_argument(
        "--density-tol", type=float, default=0.5, help="Minimum density tolerance for layers"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # Sorting parameters
    parser.add_argument(
        "--sort-by-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use density-based sorting instead of area-based sorting.",
    )

    # Visualization parameters
    parser.add_argument(
        "--visualize-superitems", action="store_true", help="Visualize superitems generated"
    )
    parser.add_argument(
        "--visualize-maxrects", action="store_true", help="Visualize maxrects layers generated"
    )
    parser.add_argument(
        "--visualize-layers", action="store_true", help="Visualize layers generated"
    )
    parser.add_argument(
        "--visualize-filtered-layers",
        action="store_true",
        help="Visualize filtered layers generated",
    )
    parser.add_argument("--visualize-bins", action="store_true", help="Visualize bins generated")
    parser.add_argument("--use-vtk", action="store_true", help="Use VTK for 3D visualization")
    parser.add_argument(
        "--vtk-interactive", action="store_true", help="Show VTK in interactive window"
    )
    parser.add_argument(
        "--vtk-resolution",
        type=str,
        default="1200x900",
        help="VTK render resolution as WIDTHxHEIGHT.",
    )

    # GA visualization options
    parser.add_argument("--visualize-ga", action="store_true", help="Enable GA visualization")
    parser.add_argument(
        "--visualize-ga-initialization",
        action="store_true",
        help="Visualize GA initialization strategies",
    )
    parser.add_argument(
        "--visualize-ga-evolution", action="store_true", help="Visualize GA evolution"
    )
    parser.add_argument(
        "--visualize-ga-operations", action="store_true", help="Visualize GA operations"
    )

    # CompactBin parameters
    parser.add_argument(
        "--use-sequential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sequential bin packing approach.",
    )
    parser.add_argument(
        "--validate-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate the final bin packing solution.",
    )
    parser.add_argument("--skip-compacting", action="store_true", help="Skip the compacting phase")

    args = parser.parse_args()

    # Enable verbose logging if specified
    if args.verbose:
        logger.add("debug.log", level="DEBUG")

    # Load order data
    logger.info(f"Loading {args.ordered_products_path} products from dataset...")
    dataset = ProductDataset(args.ordered_products_path, seed=args.seed)
    order = dataset.get_full_order()
    if order.empty:
        raise ValueError(
            f"No valid products were loaded from '{args.ordered_products_path}'. Check the CSV columns and file path."
        )
    logger.info(f"Loaded an order with {len(order)} items.")

    # Performance optimization: Adjust parameters based on order size
    order_size = len(order)
    if order_size > 100:
        logger.info(
            f"Large order detected ({order_size} items). Applying performance optimizations..."
        )
        args.max_iters = min(args.max_iters, 1)
        args.density_tol = max(args.density_tol, 0.3)
        logger.info(
            f"Adjusted parameters: max_iters={args.max_iters}, density_tol={args.density_tol}"
        )

    # Create results directory once at the beginning
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file_base = Path(args.ordered_products_path).stem

    # Create VTK visualizer once if needed
    render_config = RenderConfig(
        use_vtk=args.use_vtk,
        vtk_interactive=args.vtk_interactive,
        vtk_resolution=args.vtk_resolution,
    )
    vtk_viz, vtk_width, vtk_height = (None, None, None)
    if args.use_vtk:
        vtk_viz, vtk_width, vtk_height = create_vtk_visualizer(render_config, config.PALLET_DIMS)

    # Initialize superitems pool
    final_layer_pool = layers.LayerPool(superitems.SuperitemPool(), config.PALLET_DIMS)
    working_order = order.copy()
    all_singles_removed = []

    # Time to start the procedure
    time_start = time.time()

    for iter in range(args.max_iters):
        logger.info(f"MAXRECTS iteration {iter + 1}/{args.max_iters}")

        # Generate superitems and filter
        superitems_list, singles_removed = superitems.SuperitemPool.gen_superitems(
            order=working_order,
            pallet_dims=config.PALLET_DIMS,
            max_vstacked=2,
            horizontal=True,
            horizontal_type="two-width",
            visualize=args.visualize_superitems,
            sort_by_density=args.sort_by_density,
        )
        superitems_pool = superitems.SuperitemPool(superitems_list)

        all_singles_removed += singles_removed

        # Use MAXRECTS for layer generation with density-based sorting
        layer_pool = maxrects.maxrects_warm_start(
            superitems_pool,
            height_tol=10,
            density_tol=args.density_tol,
            add_single=True,
            visualize=args.visualize_maxrects,
            sort_by_density=args.sort_by_density,
        )

        if args.visualize_layers:
            visualization.plot_layers(layer_pool, config.PALLET_DIMS)

        # Filter and finalize
        layer_pool = layer_pool.filter_layers(
            min_density=args.density_tol,
            two_dims=False,
            max_coverage_all=3,
            max_coverage_single=3,
            visualize_filtered_layers=args.visualize_filtered_layers,
        )

        # Add layers to final layer pool
        final_layer_pool.extend(layer_pool)
        item_coverage = final_layer_pool.item_coverage()
        not_covered = [k for k, v in item_coverage.items() if not v]

        # Early termination conditions
        if len(not_covered) == len(working_order):
            logger.info("Stopping as no improvement is observed.")
            break

        # For large orders, limit iterations to prevent timeouts
        if order_size > 100 and iter >= 0:
            logger.info("Stopping early for large order to prevent timeout.")
            break

        if "productid" in order.columns:
            working_order = order[order["productid"].astype(str).isin(not_covered)].copy()
        else:
            working_order = order.loc[order.index.astype(str).isin(not_covered)].copy()

    # Initial bin creation (Pre-GA stage)
    logger.info("Building initial bins with standard BinPool...")
    bin_pool = bins.BinPool(final_layer_pool, config.PALLET_DIMS)

    # Store the original bin data for visualization comparison
    initial_bin_data = bin_pool.to_dataframe() if len(bin_pool) > 0 else None

    # KPI Analysis for Pre-GA stage (MAXRECTS only)
    pre_ga_scores = evaluate_stage_kpi(initial_bin_data, "Pre-GA (MAXRECTS)", config.PALLET_DIMS)

    # Check for unplaced items that need optimization
    not_covered_superitems = bin_pool.layer_pool.not_covered_superitems()

    if not_covered_superitems:
        # Compute covered_ids after filtering to avoid duplicates
        covered_ids = set()
        for layer in final_layer_pool:
            covered_ids.update(layer.get_unique_items_ids())

        # Filter to get unique residuals
        unique_residuals = []
        seen_ids = set()

        for s in not_covered_superitems:
            ids = s.id if isinstance(s.id, list) else [s.id]
            # Only include if none of its items are already covered or seen
            if not any(id in covered_ids or id in seen_ids for id in ids):
                unique_residuals.append(s)
                seen_ids.update(ids)

        # Performance optimization: Limit residual items for large orders
        if order_size > 100 and len(unique_residuals) > 20:
            logger.info(
                f"Large order detected. Limiting residual items from {len(unique_residuals)} to 20 for performance."
            )
            unique_residuals = unique_residuals[:20]

        logger.info(f"Running GA to pack {len(unique_residuals)} unique residual items")

        # GA visualization options
        ga_visualization_options = {
            "enable": args.visualize_ga,
            "init": args.visualize_ga_initialization,
            "evolution": args.visualize_ga_evolution,
            "operations": args.visualize_ga_operations,
            "crossover": args.visualize_ga_operations,  # Enable crossover visualization
            "mutation": args.visualize_ga_operations,  # Enable mutation visualization
        }

        # Before running GA optimization, calculate max z-level from MAXRECTS
        max_z_level = 0
        if final_layer_pool:
            # Find the maximum z-level used by MAXRECTS layers
            for layer in final_layer_pool:
                # Fixed: get_items_coords() returns a dictionary, not a list
                items_coords = layer.get_items_coords()
                for item_id, coord in items_coords.items():
                    # Need to look up the corresponding superitem by item_id
                    for superitem in layer.superitems_pool:
                        if item_id in superitem.id:
                            item_height = superitem.height
                            max_z_level = max(max_z_level, coord.z + item_height)
                            break

        # Run GA optimization for residuals with timeout protection
        try:
            ga_layer_pool = ga_optimizer.optimize_residuals(
                unique_residuals,
                final_layer_pool,
                config.PALLET_DIMS,
                base_z_level=max_z_level,
                visualization_options=ga_visualization_options,
            )

            # Add GA-optimized layers to final layer pool
            if ga_layer_pool and len(ga_layer_pool) > 0:
                final_layer_pool.extend(ga_layer_pool)
                logger.info(f"GA optimization added {len(ga_layer_pool)} new layers")
            else:
                logger.info("GA optimization completed but no new layers were added")

        except Exception as e:
            logger.warning(f"GA optimization failed: {str(e)}. Continuing without GA optimization.")
            ga_layer_pool = None

        # Rebuild bins using updated layer pool
        logger.info("Rebuilding bins with GA-optimized layers...")
        bin_pool = bins.BinPool(final_layer_pool, config.PALLET_DIMS)

    # Store the intermediate bin data after GA but before compacting
    intermediate_bin_data = bin_pool.to_dataframe() if len(bin_pool) > 0 else None

    # KPI Analysis after GA but before compacting
    post_ga_scores = evaluate_stage_kpi(intermediate_bin_data, "Post-GA", config.PALLET_DIMS)

    # Apply compacting for improved placement if not skipped
    compact_bin_pool = None
    if not args.skip_compacting:
        logger.info("Applying bin compacting and optimization...")
        compact_time_start = time.time()

        compact_bin_pool = bins.CompactBinPool(
            bin_pool, use_sequential=args.use_sequential, validate_final=args.validate_final
        )

        compact_time_end = time.time()
        compact_time = compact_time_end - compact_time_start
        logger.info(f"Post-processing time: {compact_time:.4f} seconds")

        # Report validation summary if available
        if hasattr(compact_bin_pool, "validation_summary") and compact_bin_pool.validation_summary:
            summary = compact_bin_pool.validation_summary
            logger.info(
                f"Validation summary: {summary['valid_items']}/{summary['total_items']} items valid ({summary['valid_percentage']:.2f}%)"
            )

            if summary["removed_items"] > 0:
                logger.warning(
                    f"Removed {summary['removed_items']} items: {summary['overlapping_items_count']} overlapping, {summary['insufficient_support_count']} insufficient support"
                )
                logger.warning(
                    f"Issues found in {summary['bins_with_issues']}/{len(compact_bin_pool.compact_bins)} bins"
                )
    else:
        logger.info("Skipping bin compacting phase as requested.")

    final_bin_pool = compact_bin_pool if compact_bin_pool else bin_pool

    if isinstance(final_bin_pool, bins.CompactBinPool):
        df = final_bin_pool.to_dataframe(use_validated=args.validate_final)
    else:
        df = final_bin_pool.to_dataframe()

    # KPI Analysis after compacting
    post_compact_scores = evaluate_stage_kpi(df, "Final (Post-Compact)", config.PALLET_DIMS)

    if df is not None and not df.empty:
        # Create comprehensive KPI comparison across all stages
        if pre_ga_scores and post_ga_scores and post_compact_scores:
            metrics = []
            pre_ga_values = []
            post_ga_values = []
            post_compact_values = []

            logger.info("Comprehensive KPI Comparison (Pre-GA vs Post-GA vs Final):")

            # Fill metrics and values
            for metric in pre_ga_scores.keys():
                if metric in ["n_unpacked", "max_stacking_height"]:
                    continue
                pre_val = pre_ga_scores.get(metric)
                post_val = post_ga_scores.get(metric)
                final_val = post_compact_scores.get(metric)
                if (
                    pre_val is not None
                    and post_val is not None
                    and final_val is not None
                    and isinstance(pre_val, (int, float))
                    and isinstance(post_val, (int, float))
                    and isinstance(final_val, (int, float))
                ):
                    metrics.append(metric)
                    pre_ga_values.append(pre_val)
                    post_ga_values.append(post_val)
                    post_compact_values.append(final_val)

            # KPI comparison - console output only (no matplotlib charts)
            if not metrics:
                logger.warning("No valid numeric metrics found")
                return

            # Calculate improvements
            overall_pre_ga = pre_ga_scores["overall_score"]
            overall_post_ga = post_ga_scores["overall_score"]
            overall_final = post_compact_scores["overall_score"]

            ga_improvement = overall_post_ga - overall_pre_ga
            compact_improvement = overall_final - overall_post_ga
            total_improvement = overall_final - overall_pre_ga

            # Print summary to console
            logger.info("\nComprehensive KPI Performance Summary:")
            logger.info("=" * 50)
            logger.info(f"Overall KPI Score Evolution:")
            logger.info(f"  Pre-GA (MAXRECTS):     {overall_pre_ga:.4f}")
            logger.info(f"  Post-GA:               {overall_post_ga:.4f} ({ga_improvement:+.4f})")
            logger.info(
                f"  Final (Post-Compact):  {overall_final:.4f} ({compact_improvement:+.4f})"
            )
            logger.info(
                f"  Total Improvement:     {total_improvement:+.4f} ({total_improvement/overall_pre_ga*100:+.2f}%)"
            )

            # Show which stage contributed most to improvement
            if abs(ga_improvement) > abs(compact_improvement):
                logger.info(
                    f"\nGreatest improvement came from: GA Optimization ({ga_improvement:+.4f})"
                )
            elif abs(compact_improvement) > abs(ga_improvement):
                logger.info(
                    f"\nGreatest improvement came from: Compacting ({compact_improvement:+.4f})"
                )
            else:
                logger.info(f"\nGA and Compacting contributed equally to improvement")

        elif pre_ga_scores and post_ga_scores:
            # Handle case where compacting was skipped
            logger.info("KPI Comparison (Pre-GA vs Post-GA, no compacting):")

            for metric in post_ga_scores:
                if metric in pre_ga_scores and metric != "n_unpacked":  # Skip raw count
                    if post_ga_scores[metric] is not None and pre_ga_scores[metric] is not None:
                        change = post_ga_scores[metric] - pre_ga_scores[metric]
                        pct_change = (
                            (change / pre_ga_scores[metric] * 100)
                            if pre_ga_scores[metric] != 0
                            else float("inf")
                        )

                        logger.info(
                            f"{metric}: {pre_ga_scores[metric]:.4f} to {post_ga_scores[metric]:.4f} "
                            + f"(Change: {change:+.4f}, {pct_change:+.2f}%)"
                        )
                    else:
                        logger.info(
                            f"{metric}: {pre_ga_scores[metric] if pre_ga_scores[metric] is not None else 'N/A'} to {post_ga_scores[metric] if post_ga_scores[metric] is not None else 'N/A'}"
                        )
    else:
        logger.warning("No final bin data available for final KPI analysis")

    # Calculate basic statistics
    packed_items_count = len(set(df["item"])) if df is not None and not df.empty else 0
    total_items_count = len(order)
    packing_rate = (packed_items_count / total_items_count * 100) if total_items_count else 0

    # Time to end the procedure
    time_end = time.time()
    time_elapsed = time_end - time_start

    # Save result files

    df.to_csv(results_dir / f"packed_items_{output_file_base}.csv", index=False)

    # Output the algorithm's result in JSON format using stable source identifiers
    if "productid" in order.columns:
        order_lookup = order.copy()
        order_lookup["productid"] = order_lookup["productid"].astype(str)
        order_lookup = order_lookup.drop_duplicates(subset=["productid"]).set_index("productid")
    else:
        order_lookup = order.copy()
        order_lookup.index = order_lookup.index.astype(str)

    if "order_id" in order.columns and not order["order_id"].isna().all():
        order_id = str(order["order_id"].iloc[0])
    else:
        order_id = os.path.splitext(os.path.basename(args.ordered_products_path))[0]

    output_json = {order_id: []}
    for _, row in df.iterrows():
        item_id = str(row["item"])
        try:
            order_row = order_lookup.loc[item_id]
        except Exception:
            order_row = None

        if order_row is None:
            logger.warning(
                f"Skipping JSON export for item '{item_id}' because it was not found in the source order."
            )
            continue

        item_dict = {
            "article": order_row["article"] if "article" in order_row else str(item_id),
            "id": item_id,
            "product_group": order_row["product_group"] if "product_group" in order_row else None,
            "length/mm": int(row["length"]),
            "width/mm": int(row["width"]),
            "height/mm": int(row["height"]),
            "weight/kg": float(order_row["weight"]) if "weight" in order_row else None,
            "sequence": int(order_row["sequence"]) if "sequence" in order_row else None,
        }
        output_json[order_id].append(
            {
                "item": item_dict,
                "flb_coordinates": [int(row["y"]), int(row["x"]), int(row["z"])],
                "orientation": 0,
            }
        )
    # Save the JSON
    utils.save_json(output_json, results_dir / f"packed_items_{output_file_base}.json")

    # Display and save visualization
    if args.visualize_bins:
        logger.info("Creating visualizations...")

        # Create progression visualization first (logical order: Stage 1 -> Stage 2 -> Final)
        if args.use_vtk and vtk_viz is not None:
            try:
                # Visualize Stage 1: MAXRECTS
                visualize_stage_vtk(
                    vtk_viz,
                    initial_bin_data,
                    "Stage 1: MAXRECTS Baseline",
                    f"vtk_stage1_maxrects_{output_file_base}.png",
                    render_config,
                    vtk_width,
                    vtk_height,
                )

                # Visualize Stage 2: After GA
                visualize_stage_vtk(
                    vtk_viz,
                    intermediate_bin_data,
                    "Stage 2: After GA Optimization",
                    f"vtk_stage2_ga_{output_file_base}.png",
                    render_config,
                    vtk_width,
                    vtk_height,
                )
            except Exception as e:
                logger.error(f"VTK progression visualization failed: {e}")

        # Visualize final solution last
        visualize_bins_solution(
            df,
            render_config,
            config.PALLET_DIMS,
            "Final_Solution",
            output_file_base,
            vtk_viz,
            vtk_width,
            vtk_height,
        )

    # Display completion message
    logger.info("Processing completed successfully!")
    logger.info("Final Bin Utilization Summary:")
    logger.info("----------------------------------")
    logger.info(f"Total time elapsed: {time_elapsed:.2f} seconds")
    logger.info(f"Items packed: {packed_items_count}/{total_items_count} ({packing_rate:.2f}%)")


if __name__ == "__main__":
    main()

# Example usage:
# Basic execution with default parameters:
# python -m src.main --ordered-products-path data/1.csv --visualize-bins
#
# VTK visualization:
# python -m src.main --ordered-products-path data/1.csv --visualize-bins --use-vtk
#
# Interactive VTK visualization:
# python -m src.main --ordered-products-path data/1.csv --visualize-bins --use-vtk --vtk-interactive
#
# Full pipeline with GA visualization and verbose logging:
# python -m src.main --ordered-products-path data/5.csv --seed 42 --max-iters 1 \
#     --density-tol 0.5 --verbose --visualize-bins --visualize-ga --visualize-ga-evolution
#
