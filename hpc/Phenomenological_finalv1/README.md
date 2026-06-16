

## Plot topology raster
#### single topology, everything in topo_dir/figures/
python plot_topology_rasters.py campaign_cadex_hhgap_v1/sweep_egeos_task0000/topo_00042

#### custom output, first 20 iters only, 6-column overview grid
python plot_topology_rasters.py topo_00042 --out_dir ./figs --max-iters 20 --n-cols 6

#### from Python (e.g. inside a notebook)
from plot_topology_rasters import plot_topology_rasters
saved = plot_topology_rasters("topo_00042", dpi=200, overview=True)
