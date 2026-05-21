from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


@dataclass(frozen=True)
class OmeTiffInfo:
    path: Path
    shape: tuple[int, ...]
    axes: str
    dtype: str
    size_c: int
    size_z: int
    size_y: int
    size_x: int
    physical_size_x_um: float
    physical_size_y_um: float
    physical_size_z_um: float
    channel_names: tuple[str | None, ...]


def _transpose_to_canonical(data: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    axes = axes.upper().replace("S", "C")
    if "Q" in axes and "C" not in axes and axes.count("Q") == 1:
        axes = axes.replace("Q", "C")

    supported_axes = set("TCZYX")
    squeeze_axes = tuple(
        i for i, axis in enumerate(axes)
        if axis not in supported_axes and data.shape[i] == 1
    )
    if squeeze_axes:
        data = np.squeeze(data, axis=squeeze_axes)
        axes = "".join(axis for i, axis in enumerate(axes) if i not in squeeze_axes)

    unsupported = [axis for axis in axes if axis not in supported_axes]
    if unsupported:
        raise ValueError(f"Unsupported OME-TIFF axes {axes}: {unsupported}")

    canonical_axes = "".join(axis for axis in "TCZYX" if axis in axes)
    if axes != canonical_axes:
        data = np.transpose(data, tuple(axes.index(axis) for axis in canonical_axes))
    return data, canonical_axes


def _ome_namespace(root: ET.Element) -> dict[str, str]:
    if root.tag.startswith("{"):
        return {"ome": root.tag.split("}")[0].strip("{")}
    return {}


def _find_pixels(root: ET.Element) -> ET.Element | None:
    ns = _ome_namespace(root)
    if ns:
        return root.find(".//ome:Pixels", ns)
    return root.find(".//Pixels")


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def inspect_ome_tiff(path: str | Path) -> OmeTiffInfo:
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(v) for v in series.shape)
        axes = series.axes
        dtype = str(series.dtype)

        physical_x = physical_y = physical_z = 1.0
        channel_names: tuple[str | None, ...] = tuple()
        if tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            pixels = _find_pixels(root)
            if pixels is not None:
                physical_x = _parse_float(pixels.get("PhysicalSizeX"), physical_x)
                physical_y = _parse_float(pixels.get("PhysicalSizeY"), physical_y)
                physical_z = _parse_float(pixels.get("PhysicalSizeZ"), physical_z)

                ns = _ome_namespace(root)
                channels = pixels.findall("ome:Channel", ns) if ns else pixels.findall("Channel")
                channel_names = tuple(ch.get("Name") for ch in channels)

    axis_sizes = dict(zip(axes.upper().replace("S", "C"), shape))
    return OmeTiffInfo(
        path=path,
        shape=shape,
        axes=axes,
        dtype=dtype,
        size_c=int(axis_sizes.get("C", 1)),
        size_z=int(axis_sizes.get("Z", 1)),
        size_y=int(axis_sizes.get("Y", 1)),
        size_x=int(axis_sizes.get("X", 1)),
        physical_size_x_um=physical_x,
        physical_size_y_um=physical_y,
        physical_size_z_um=physical_z,
        channel_names=channel_names,
    )


def _normalise_crop(
    crop: tuple[slice, slice, slice] | None,
    zyx_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    if crop is None:
        return (slice(None), slice(None), slice(None))
    if len(crop) != 3:
        raise ValueError("Crop must be a tuple of three slices in Z, Y, X order.")
    normalised = []
    for axis_slice, axis_len in zip(crop, zyx_shape):
        start, stop, step = axis_slice.indices(axis_len)
        normalised.append(slice(start, stop, step))
    return tuple(normalised)  # type: ignore[return-value]


def load_ome_tiff_czyx(
    path: str | Path,
    crop: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, str, OmeTiffInfo]:
    """Load an OME-TIFF into CZYX order.

    This is useful for validation and cropped tests. Full SIM stacks can be
    several GB when decompressed; prefer load_ome_tiff_channel_zyx for analysis.
    """
    info = inspect_ome_tiff(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        data, axes = _transpose_to_canonical(data, series.axes)

    if "T" in axes:
        data = np.take(data, 0, axis=axes.index("T"))
        axes = axes.replace("T", "", 1)
    if axes == "ZYX":
        data = data[np.newaxis, :, :, :]
        axes = "CZYX"
    if axes != "CZYX":
        raise ValueError(f"Expected OME-TIFF to reduce to CZYX, got {axes} {data.shape}")

    z_crop, y_crop, x_crop = _normalise_crop(crop, data.shape[1:])
    data = data[:, z_crop, y_crop, x_crop]
    return data.astype(np.uint16, copy=False), axes, info


def load_ome_tiff_channel_zyx(
    path: str | Path,
    channel: int,
    crop: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, OmeTiffInfo]:
    """Load one channel from an OME-TIFF as ZYX.

    For planar OME-TIFFs this reads only the pages for the requested channel.
    If the TIFF is stored in a layout that cannot be addressed page-by-page, it
    falls back to the full CZYX loader.
    """
    info = inspect_ome_tiff(path)
    if channel < 0 or channel >= info.size_c:
        raise IndexError(f"Channel {channel} is out of bounds for {info.size_c} channels.")

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper().replace("S", "C")
        shape = tuple(int(v) for v in series.shape)
        if "T" in axes and shape[axes.index("T")] > 1:
            raise ValueError("Timelapse OME-TIFFs are not supported for 3D SODA yet.")

        if set(axes).issuperset({"C", "Z", "Y", "X"}) and len(tif.pages) >= info.size_c * info.size_z:
            non_spatial_axes = [axis for axis in axes if axis not in {"Y", "X"}]
            non_spatial_shape = [shape[axes.index(axis)] for axis in non_spatial_axes]
            template = tif.pages[0].asarray()
            out = np.empty((info.size_z, template.shape[-2], template.shape[-1]), dtype=template.dtype)

            for z in range(info.size_z):
                coords = []
                for axis in non_spatial_axes:
                    if axis == "T":
                        coords.append(0)
                    elif axis == "C":
                        coords.append(channel)
                    elif axis == "Z":
                        coords.append(z)
                    else:
                        coords.append(0)
                page_index = int(np.ravel_multi_index(tuple(coords), tuple(non_spatial_shape), order="C"))
                out[z] = tif.pages[page_index].asarray()

            z_crop, y_crop, x_crop = _normalise_crop(crop, out.shape)
            return out[z_crop, y_crop, x_crop].astype(np.uint16, copy=False), info

    data, _, info = load_ome_tiff_czyx(path, crop=crop)
    return data[channel].astype(np.uint16, copy=False), info


def parse_crop(value: str | None) -> tuple[slice, slice, slice] | None:
    if value is None or value.strip() == "":
        return None
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("Crop must be formatted as z0:z1,y0:y1,x0:x1")
    slices = []
    for part in parts:
        start_stop = part.split(":")
        if len(start_stop) != 2:
            raise ValueError("Crop must be formatted as z0:z1,y0:y1,x0:x1")
        start = int(start_stop[0]) if start_stop[0] else None
        stop = int(start_stop[1]) if start_stop[1] else None
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]
