from typing import Literal

import numpy as np
import pyvista as pv
from jaxtyping import Float

from ..preprocessing.bbox import BoundingSquare


def render_pointcloud(
    plotter: pv.Plotter,
    pointcloud: Float[np.ndarray, "N 3"],
    color: str,
    point_size: float = 1.0,
    opacity: float = 0.2,
    background_color: str = "black",
    projection: Literal["xy", "yz", "zx", "yx", "zy", "xz", None] = "xy",
) -> pv.Plotter:
    plotter.set_background(color=background_color)
    plotter.add_mesh(
        pointcloud,
        color=color,
        style="points_gaussian",
        point_size=point_size,
        opacity=opacity,
    )
    if projection is not None:
        proj_fn = getattr(plotter, f"view_{projection}")
        proj_fn()
        plotter.enable_parallel_projection()

    return plotter


def render_boundingsquare(
    plotter: pv.Plotter,
    bbox: BoundingSquare,
    color: str = "white",
    line_width: float = 1.5,
    opacity: float = 1.0,
    z_height: float = 0.0,
) -> pv.Plotter:
    rect = pv.Rectangle(
        [
            [bbox.x0, bbox.y0, z_height],
            [bbox.x1, bbox.y1, z_height],
            [bbox.x0, bbox.y1, z_height],
        ]
    )
    plotter.add_mesh(
        rect,
        show_edges=True,
        style="wireframe",
        color=color,
        edge_color=color,
        line_width=line_width,
        opacity=opacity,
    )
    return plotter
