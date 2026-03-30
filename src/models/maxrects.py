import numpy as np
from loguru import logger
from rectpack import SORT_AREA, PackingBin, PackingMode, newPacker
from rectpack.maxrects import MaxRectsBaf, MaxRectsBl, MaxRectsBlsf, MaxRectsBssf

from src import config
from src.models import layers, superitems
from src.utils import utils, visualization

MAXRECTS_PACKING_STRATEGIES = [MaxRectsBaf, MaxRectsBssf, MaxRectsBlsf, MaxRectsBl]


def maxrects_multiple_layers(
    superitems_pool, pallet_dims, add_single=True, visualize=False, sort_by_density=True
):
    """Generate layers using MAXRECTS algorithm."""
    logger.debug("MR-ML-Offline starting")

    if add_single:
        # Build initial layer pool
        layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=add_single)

        # Create the maxrects packing algorithm with density-based sorting
        if sort_by_density:
            # Calculate density for each superitem (volume / area)
            densities = []
            for s in superitems_pool:
                volume = s.volume
                area = s.width * s.length
                density = volume / area if area > 0 else 0
                densities.append(density)

            # Sort superitems by density (highest first)
            sorted_indices = utils.argsort(densities, reverse=True)
            sorted_superitems = [superitems_pool[i] for i in sorted_indices]

            # Create packer with area sorting (will be overridden by our custom order)
            packer = newPacker(
                mode=PackingMode.Offline,
                bin_algo=PackingBin.Global,
                pack_algo=MAXRECTS_PACKING_STRATEGIES[0],  # Use first strategy
                sort_algo=SORT_AREA,
                rotation=False,
            )

            # Add bins and items in density-sorted order
            packer.add_bin(pallet_dims.width, pallet_dims.length, count=float("inf"))

            # Add superitems in density-sorted order
            for i, s in enumerate(sorted_superitems):
                packer.add_rect(s.width, s.length, rid=sorted_indices[i])

            logger.info(
                f"Using density-based sorting for MAXRECTS (density range: {min(densities):.2f}-{max(densities):.2f})"
            )
        else:
            # Original area-based approach
            packer = newPacker(
                mode=PackingMode.Offline,
                bin_algo=PackingBin.Global,
                pack_algo=MAXRECTS_PACKING_STRATEGIES[0],
                sort_algo=SORT_AREA,
                rotation=False,
            )

            packer.add_bin(pallet_dims.width, pallet_dims.length, count=float("inf"))

            # Add superitems in original order
            ws, ds, _ = superitems_pool.get_superitems_dims()
            for i, (w, d) in enumerate(zip(ws, ds)):
                packer.add_rect(w, d, rid=i)

        # Start the packing procedure
        packer.pack()

        # Build a layer pool
        for layer in packer:
            spool, scoords = [], []
            for superitem in layer:
                spool += [superitems_pool[superitem.rid]]
                scoords += [utils.Coordinate(superitem.x, superitem.y)]

            spool = superitems.SuperitemPool(superitems=spool)
            layer_pool.add(layers.Layer(spool, scoords, pallet_dims))
            layer_pool.sort_by_densities(two_dims=False)

        uncovered = len(layer_pool.not_covered_superitems())
    else:
        generated_pools = []
        for strategy in MAXRECTS_PACKING_STRATEGIES:
            # Build initial layer pool
            layer_pool = layers.LayerPool(superitems_pool, pallet_dims, add_single=add_single)

            # Create the maxrects packing algorithm with density-based sorting option
            if sort_by_density:
                # Calculate density for each superitem
                densities = []
                for s in superitems_pool:
                    volume = s.volume
                    area = s.width * s.length
                    density = volume / area if area > 0 else 0
                    densities.append(density)

                # Sort superitems by density (highest first)
                sorted_indices = utils.argsort(densities, reverse=True)
                sorted_superitems = [superitems_pool[i] for i in sorted_indices]

                packer = newPacker(
                    mode=PackingMode.Offline,
                    bin_algo=PackingBin.Global,
                    pack_algo=strategy,
                    sort_algo=SORT_AREA,
                    rotation=False,
                )

                packer.add_bin(pallet_dims.width, pallet_dims.length, count=float("inf"))

                # Add superitems in density-sorted order
                for i, s in enumerate(sorted_superitems):
                    packer.add_rect(s.width, s.length, rid=sorted_indices[i])
            else:
                # Original area-based approach
                packer = newPacker(
                    mode=PackingMode.Offline,
                    bin_algo=PackingBin.Global,
                    pack_algo=strategy,
                    sort_algo=SORT_AREA,
                    rotation=False,
                )

                packer.add_bin(pallet_dims.width, pallet_dims.length, count=float("inf"))

                # Add superitems to be packed
                ws, ds, _ = superitems_pool.get_superitems_dims()
                for i, (w, d) in enumerate(zip(ws, ds)):
                    packer.add_rect(w, d, rid=i)

            # Start the packing procedure
            packer.pack()

            # Build a layer pool
            for layer in packer:
                spool, scoords = [], []
                for superitem in layer:
                    spool += [superitems_pool[superitem.rid]]
                    scoords += [utils.Coordinate(superitem.x, superitem.y)]

                spool = superitems.SuperitemPool(superitems=spool)
                layer_pool.add(layers.Layer(spool, scoords, pallet_dims))
                layer_pool.sort_by_densities(two_dims=False)

            # Add the layer pool to the list of generated pools
            generated_pools += [layer_pool]

        # Find the best layer pool by considering the number of placed superitems,
        # the number of generated layers and the density of each layer dense
        uncovered = [len(pool.not_covered_superitems()) for pool in generated_pools]
        n_layers = [len(pool) for pool in generated_pools]
        densities = [pool[0].get_density(two_dims=False) for pool in generated_pools]
        pool_indexes = utils.argsort(list(zip(uncovered, n_layers, densities)), reverse=True)
        layer_pool = generated_pools[pool_indexes[0]]
        uncovered = uncovered[pool_indexes[0]]

    logger.debug(
        f"MR-ML-Offline generated {len(layer_pool)} layers with 3D densities {layer_pool.get_densities(two_dims=False)}"
    )
    logger.debug(
        f"MR-ML-Offline placed {len(superitems_pool) - uncovered}/{len(superitems_pool)} superitems"
    )

    if visualize:
        visualization.plot_layers(
            layer_pool,
            pallet_dims,
            cols=6,
            title="MAXRECTS warm_start\n (before Column Generation)",
        )

    return layer_pool


def maxrects_single_layer_offline(superitems_pool, pallet_dims, superitems_in_layer=None):
    """Pack specified superitems into a single layer using MAXRECTS."""
    logger.debug("MR-SL-Offline starting")

    # Set all superitems in layer
    if superitems_in_layer is None:
        superitems_in_layer = np.arange(len(superitems_pool))

    logger.debug(f"MR-SL-Offline {superitems_in_layer}/{len(superitems_pool)} superitems to place")

    # Iterate over each placement strategy
    ws, ds, _ = superitems_pool.get_superitems_dims()
    for strategy in MAXRECTS_PACKING_STRATEGIES:
        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Offline,
            bin_algo=PackingBin.Global,
            pack_algo=strategy,
            sort_algo=SORT_AREA,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.length, count=1)

        # Add superitems to be packed
        for i in superitems_in_layer:
            packer.add_rect(ws[i], ds[i], rid=i)

        # Start the packing procedure
        packer.pack()

        # Feasible packing with a single layer
        if len(packer) == 1 and len(packer[0]) == len(superitems_in_layer):
            spool = superitems.SuperitemPool(superitems=[superitems_pool[s.rid] for s in packer[0]])
            layer = layers.Layer(
                spool, [utils.Coordinate(s.x, s.y) for s in packer[0]], pallet_dims
            )
            logger.debug(
                f"MR-SL-Offline generated a new layer with {len(layer)} superitems "
                f"and {layer.get_density(two_dims=False)} 3D density"
            )
            return layer

    return None


def maxrects_single_layer_online(superitems_pool, pallet_dims, superitems_duals=None):
    """Pack superitems into a single layer using online MAXRECTS with priority ordering."""
    logger.debug("MR-SL-Online starting")

    # If no duals are given use superitems' heights as a fallback
    ws, ds, hs = superitems_pool.get_superitems_dims()
    if superitems_duals is None:
        superitems_duals = np.array(hs)

    # Sort rectangles by duals
    indexes = utils.argsort(list(zip(superitems_duals, hs)), reverse=True)
    logger.debug(
        f"MR-SL-Online {sum(superitems_duals[i] > 0 for i in indexes)} non-zero duals to place"
    )

    # Iterate over each placement strategy
    generated_layers, num_duals = [], []
    for strategy in MAXRECTS_PACKING_STRATEGIES:
        # Create the maxrects packing algorithm
        packer = newPacker(
            mode=PackingMode.Online,
            pack_algo=strategy,
            rotation=False,
        )

        # Add one bin representing one layer
        packer.add_bin(pallet_dims.width, pallet_dims.length, count=1)

        # Online packing procedure
        n_packed, non_zero_packed, layer_height = 0, 0, 0
        for i in indexes:
            if superitems_duals[i] > 0 or hs[i] <= layer_height:
                packer.add_rect(ws[i], ds[i], i)
                if len(packer[0]) > n_packed:
                    n_packed = len(packer[0])
                    if superitems_duals[i] > 0:
                        non_zero_packed += 1
                    if hs[i] > layer_height:
                        layer_height = hs[i]
        num_duals += [non_zero_packed]

        # Build layer after packing
        spool, coords = [], []
        for s in packer[0]:
            spool += [superitems_pool[s.rid]]
            coords += [utils.Coordinate(s.x, s.y)]
        layer = layers.Layer(superitems.SuperitemPool(spool), coords, pallet_dims)
        generated_layers += [layer]

    # Find the best layer by taking into account the number of
    # placed superitems with non-zero duals and density
    layer_indexes = utils.argsort(
        [
            (duals, layer.get_density(two_dims=False))
            for duals, layer in zip(num_duals, generated_layers)
        ],
        reverse=True,
    )
    layer = generated_layers[layer_indexes[0]]

    logger.debug(
        f"MR-SL-Online generated a new layer with {len(layer)} superitems "
        f"(of which {num_duals[layer_indexes[0]]} with non-zero dual) "
        f"and {layer.get_density(two_dims=False)} 3D density"
    )
    return layer


def maxrects_warm_start(
    superitems_pool,
    height_tol=0,
    density_tol=0.5,
    add_single=False,
    visualize=False,
    sort_by_density=True,
):
    """Generate initial layers using MAXRECTS with height grouping."""
    logger.info("MR computing layers")

    # Compute height groups and initialize initial layer pool
    height_groups = utils.get_height_groups(
        superitems_pool, config.PALLET_DIMS, height_tol=height_tol, density_tol=density_tol
    )
    # If no height groups are identified fallback to one group
    if len(height_groups) == 0:
        logger.debug(f"MR found no height groups, falling back to standard procedure")
        height_groups = [superitems_pool]
    # Initial empty layer pool
    mr_layer_pool = layers.LayerPool(superitems_pool, config.PALLET_DIMS)

    # Call maxrects for each height group and merge all the layer pools
    for i, spool in enumerate(height_groups):
        logger.info(f"MR processing height group {i + 1}/{len(height_groups)}")
        layer_pool = maxrects_multiple_layers(
            spool,
            config.PALLET_DIMS,
            add_single=add_single,
            visualize=visualize,
            sort_by_density=sort_by_density,
        )
        mr_layer_pool.extend(layer_pool)
    logger.info(f"MR generated {len(mr_layer_pool)} layers")

    # Return the final layer pool
    return mr_layer_pool
