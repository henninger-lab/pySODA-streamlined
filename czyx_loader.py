from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


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
        raise ValueError(f"Unsupported image axes {axes}: {unsupported}")

    canonical_axes = "".join(axis for axis in "TCZYX" if axis in axes)
    if axes != canonical_axes:
        data = np.transpose(data, tuple(axes.index(axis) for axis in canonical_axes))
    return data, canonical_axes


def _normalise_channels(channels: list[int] | tuple[int, ...] | None, size_c: int) -> list[int]:
    if channels is None:
        channels = list(range(size_c))
    else:
        channels = list(channels)
    if not channels:
        raise ValueError("At least one channel must be selected.")
    invalid = [channel for channel in channels if channel < 0 or channel >= size_c]
    if invalid:
        raise ValueError(f"Invalid channel(s) {invalid}; available range is 0-{size_c - 1}.")
    return channels


def load_tiff_czyx(
    path: str | Path,
    channels: list[int] | tuple[int, ...] | None = None,
    time_index: int = 0,
) -> np.ndarray:
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper().replace("S", "C")
        shape = tuple(int(v) for v in series.shape)
        axis_sizes = dict(zip(axes, shape))
        if (
            set(axes).issuperset({"C", "Z", "Y", "X"})
            and len(tif.pages) >= int(axis_sizes.get("C", 1)) * int(axis_sizes.get("Z", 1))
        ):
            selected = _normalise_channels(channels, int(axis_sizes["C"]))
            volumes = [_load_tiff_channel_zyx_pages(tif, axes, shape, channel, time_index) for channel in selected]
            return np.stack(volumes, axis=0).astype(np.uint16, copy=False)

        data = series.asarray()
        data, axes = _transpose_to_canonical(data, series.axes)

    if "T" in axes:
        t_axis = axes.index("T")
        time_index = min(time_index, data.shape[t_axis] - 1)
        data = np.take(data, time_index, axis=t_axis)
        axes = axes.replace("T", "", 1)

    if axes == "YX":
        data = data[np.newaxis, np.newaxis, :, :]
        axes = "CZYX"
    elif axes == "ZYX":
        data = data[np.newaxis, :, :, :]
        axes = "CZYX"
    elif axes == "CYX":
        data = data[:, np.newaxis, :, :]
        axes = "CZYX"

    if axes != "CZYX":
        raise ValueError(f"Expected image to reduce to CZYX, got axes {axes} and shape {data.shape}.")

    selected = _normalise_channels(channels, data.shape[0])
    return data[selected].astype(np.uint16, copy=False)


def _load_tiff_channel_zyx_pages(
    tif: tifffile.TiffFile,
    axes: str,
    shape: tuple[int, ...],
    channel: int,
    time_index: int,
) -> np.ndarray:
    non_spatial_axes = [axis for axis in axes if axis not in {"Y", "X"}]
    non_spatial_shape = [shape[axes.index(axis)] for axis in non_spatial_axes]
    size_z = int(shape[axes.index("Z")])
    template = tif.pages[0].asarray()
    out = np.empty((size_z, template.shape[-2], template.shape[-1]), dtype=template.dtype)

    for z in range(size_z):
        coords = []
        for axis in non_spatial_axes:
            if axis == "T":
                coords.append(min(time_index, shape[axes.index("T")] - 1))
            elif axis == "C":
                coords.append(channel)
            elif axis == "Z":
                coords.append(z)
            else:
                coords.append(0)
        page_index = int(np.ravel_multi_index(tuple(coords), tuple(non_spatial_shape), order="C"))
        out[z] = tif.pages[page_index].asarray()
    return out


def _czi_read_kwargs(czi, time_index: int) -> dict[str, int]:
    dims_shape = czi.get_dims_shape()[0]
    read_kwargs = {}
    for dim in ["B", "V", "S", "M", "H", "I", "R"]:
        if dim in dims_shape:
            read_kwargs[dim] = 0
    if "T" in dims_shape:
        read_kwargs["T"] = min(time_index, dims_shape["T"][1] - 1)
    return read_kwargs


def _czi_channel_zyx(czi, channel: int, time_index: int = 0) -> np.ndarray:
    dims_shape = czi.get_dims_shape()[0]
    n_z = dims_shape.get("Z", (0, 1))[1]
    image, _ = czi.read_image(C=channel, **_czi_read_kwargs(czi, time_index))
    data = np.squeeze(image)

    while data.ndim > 3:
        data = np.squeeze(data, axis=0)
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    if data.ndim == 3:
        if n_z <= 1 and data.shape[0] != 1:
            return data[np.newaxis, :, :]
        return data
    raise ValueError(f"Unexpected CZI channel shape for channel {channel}: {data.shape}")


def load_czi_czyx(
    path: str | Path,
    channels: list[int] | tuple[int, ...] | None = None,
    time_index: int = 0,
) -> np.ndarray:
    try:
        import aicspylibczi
    except ImportError as exc:
        raise ImportError(
            "CZI loading requires aicspylibczi. Install/use the cellpose-env environment."
        ) from exc

    path = Path(path)
    czi = aicspylibczi.CziFile(path)
    dims_shape = czi.get_dims_shape()[0]
    size_c = dims_shape.get("C", (0, 1))[1]
    selected = _normalise_channels(channels, size_c)
    planes = [_czi_channel_zyx(czi, channel, time_index=time_index) for channel in selected]
    return np.stack(planes, axis=0).astype(np.uint16, copy=False)


def load_image_czyx(
    path: str | Path,
    channels: list[int] | tuple[int, ...] | None = None,
    time_index: int = 0,
) -> np.ndarray:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".czi"):
        return load_czi_czyx(path, channels=channels, time_index=time_index)
    if name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return load_tiff_czyx(path, channels=channels, time_index=time_index)
    raise ValueError(f"Unsupported image type: {path}")


def load_image_cyx_max(
    path: str | Path,
    channels: list[int] | tuple[int, ...],
    time_index: int = 0,
) -> np.ndarray:
    data = load_image_czyx(path, channels=channels, time_index=time_index)
    return np.max(data, axis=1).astype(data.dtype, copy=False)
