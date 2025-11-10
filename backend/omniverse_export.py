"""
Utility helpers for exporting pipeline profiles to Omniverse (USD) scenes.

These functions do not depend on the Omniverse Kit SDK at runtime; instead they
emit lightweight USDA files that can be loaded into Omniverse USD Composer or
Kit-based visualizers.  Each profile is represented as a `UsdGeomPoints` prim
with per-curve color coding for quick plotting/animation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CurveSpec = Dict[str, object]


def _format_point(point: Tuple[float, float, float]) -> str:
    return f"({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})"


def _format_color(color: Tuple[float, float, float]) -> str:
    return f"({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})"


def _curve_points(
    x_coords: Sequence[float],
    values: Sequence[float],
    y_offset: float,
    z_scale: float,
) -> List[Tuple[float, float, float]]:
    if len(x_coords) != len(values):
        raise ValueError("x_coords and values must be the same length.")
    return [(float(x), y_offset, float(val) * z_scale) for x, val in zip(x_coords, values)]


def export_pressure_profiles_to_usd(
    x_coords: Sequence[float],
    curves: Iterable[CurveSpec],
    output_path: Path,
    *,
    z_scale: float = 1.0,
    metadata: Dict[str, object] | None = None,
) -> Path:
    """
    Write a USD stage containing line plots for each supplied profile.

    Args:
        x_coords: Spatial grid shared by each curve.
        curves: Iterable of curve specifications. Each spec requires:
            - name (str)
            - values (Sequence[float])
            - color (Tuple[float, float, float]) in RGB 0-1
        output_path: Destination `.usda` file.
        z_scale: Optional multiplier applied to pressure values.
        metadata: Arbitrary dictionary stored as custom layer metadata.

    Returns:
        The path to the generated USD file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["#usda 1.0\n", "def Xform \"World\"\n{\n"]

    if metadata:
        lines.append("    customData = {\n")
        for key, value in metadata.items():
            lines.append(f"        string {key} = \"{value}\"\n")
        lines.append("    }\n")

    y_offset = 0.0
    for curve in curves:
        name = str(curve["name"])
        color = curve.get("color", (1.0, 1.0, 1.0))
        desc = curve.get("description", "")
        if "points" in curve:
            points_input = curve["points"]
            points = [(float(px), float(py), float(pz)) for px, py, pz in points_input]  # type: ignore[arg-type]
        else:
            values = curve["values"]
            points = _curve_points(x_coords, values, y_offset, z_scale)
        point_str = ", ".join(_format_point(pt) for pt in points)
        color_str = _format_color(color) if isinstance(color, (list, tuple)) else "(1,1,1)"

        lines.append(f"    def Points \"{name}\"\n    {{\n")
        if desc:
            lines.append(f"        string displayName = \"{desc}\"\n")
        lines.append(f"        point3f[] points = [{point_str}]\n")
        lines.append(f"        color3f[] displayColor = [{color_str}]\n")
        lines.append("        float[] widths = [0.01]\n")
        lines.append("    }\n\n")
        y_offset += 0.05

    lines.append("}\n")

    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path
