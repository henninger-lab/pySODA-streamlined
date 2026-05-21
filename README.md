# pySODA

Adapts the SODA colocalisation method from https://www.nature.com/articles/s41467-018-03053-x#Sec26 from Icy (Java) to Python.

The SODA algorithm calculates the probability that particles in super-resolution images are coupled for discrete 
intervals of distances separating them. 

**Related paper**: *Activity-dependent changes of synaptic protein clusters revealed by multidimensional analysis with multicolor STED nanoscopy*

## Requirements:

For reproducible collaborator installs, use `uv` with the committed lockfile:

```bash
uv sync --locked
```

Then run commands through the locked environment:

```bash
uv run python run_SODA_2D_pipeline.py \
  --input-dir "/path/to/full/input_images" \
  --output-dir "/path/to/pySODA_pipeline_output" \
  --channels 0 2
```

For the experimental 3D workflow, include the optional Torch dependency:

```bash
uv sync --locked --extra 3d
```

The older `requirements.txt` is kept for compatibility with the original pySODA workflow, but `pyproject.toml` plus `uv.lock` is the preferred reproducible setup.

Input images can be CZI, OME-TIFF, or regular multichannel TIFF. The selected-channel wrapper analyzes all pairwise combinations among the selected channels.

## Contents:
  - `run_SODA.py`: This is the script to execute in order to run the SODA analysis. Parameters are also set within this file.
  - `run_SODA_2D_pipeline.py`: Batch wrapper for direct CZI/OME-TIFF/TIFF loading, selected-channel inclusion, 2D max projection, pySODA execution, and combined summary tables.
  - `czyx_loader.py`: Unified CZI/OME-TIFF/TIFF loader that returns canonical `CZYX` arrays and supports selected channel inclusion.
  - `steps_SODA.py`: This file contains the SODA analysis pipeline.
  - `wavelet_SODA.py`: This file implements the base functions for the segmentation and the SODA statistical analysis.
  - `run_SODA_3D.py`: Experimental 3D OME-TIFF entrypoint for SIM stacks. This keeps Z intact, uses an existing 2D mask extruded through Z, detects spots independently in two channels, and reports physical-distance shell statistics in microns.
  - `example_image`: An example two-color STED image is provided to test the algorithm.
  - `example_output`: The pySODA output of the example image. Parameters are specified in the `output_info` text file.

## How to use pySODA:

For lab CZI/OME-TIFF folders, prefer the selected-channel batch wrapper:

```bash
uv run python run_SODA_2D_pipeline.py \
  --input-dir "/path/to/full/input_images" \
  --output-dir "/path/to/pySODA_pipeline_output" \
  --channels 0 2
```

This command loads CZI/OME-TIFF/TIFF inputs, keeps only the selected channels, max-projects Z for 2D pySODA, and writes combined CSV/XLSX summaries. See `BATCH_README.md` for tuning options.

## Sharing on GitHub

This clone tracks `https://github.com/FLClab/pySODA.git` as `origin`. For collaborative use, create a fork under your GitHub account or lab organization, push a branch containing these changes, and share that fork/branch. That keeps the upstream project intact while making this lab-specific workflow installable and citable.

The `run_soda.py` allows the user to run the SODA analysis on all images within a chosen folder using
chosen parameters.

**1 -** Edit the parameters in the `run_soda.py` file:
  
  - Directory parameters:
  
        DIRECTORY : Path containing TIF files.
        OUTPUT_DIRECTORY : Path in which to save output excel files and images. Will be created if it doesn't exist.

  - Segmentation parameters:
  
        ROI_THRESHOLD : Multiplier of ROI threshold. Higher value means more pixels considered. (float)

        CHANNEL_MASK : Channel to use as mask. This channel won't be used for SODA analysis. Set to None to generate mask from all channels.

        REMOVE_CHANNEL : Channel to remove from SODA analysis (for example, channel used for mask generation).

        
        SCALE_LIST : List of scales to be used for the multiscale product segmentation of spots.
                     Takes the form of a list containing a list of integers for each channel.
                     ex: [[1,2],   Scales for channel 0
                          [2,3]]   Scales for channel 1
                          
        SCALE_THRESHOLD : Multiplier of the wavelet transform segmentation threshold.
                          Higher value means more pixels considered. List of one value per channel.
                          ex: [1.0,   Channel 0
                               2.0]   Channel 1
  
  - SODA parameters:
                      
        MIN_SIZE : Minimum area (in pixels) of spots to be considered in the analysis. List of one value per channel.
        MIN_AXIS_LENGTH : Minimum length (in pixels) of both ellipse axes of spots to analyse. List of one value per channel.
    
        N_RINGS : Number of rings around spots (int)
        RING_WIDTH : Width of rings in pixels (int)
        SELF_SODA : Whether to compute SODA for couples of spots in the same channel as well (bool)
  
  - Output parameters:
  
        SAVE_ROI : Whether to save TIF images of spots detection and masks in OUTPUT_DIRECTORY (bool)
        WRITE_HIST : Whether to create a .pdf of the coupling probabilities by distance histogram (bool)

**2 -** Execute run_soda.py. This will run the SODA analysis on every .tif image in the chosen `DIRECTORY` using the specified
parameters.

### Examples of segmentation parameters
![Parameter examples](docs/images/param_ex.png)

## Output:

**For each TIF file**: An excel file `pySODA_[image name]_chXY.xlsx` containing information on each individual spot 
and each couple from channels X and Y.

**For the entire dataset**: An excel file `pySODA_results_[directory name].xlsx` containing global information about the analysis: coupling indices for each image, mean coupling distances, etc.

**If `SAVE_ROI` is True**: For each image, four .tif files are created.
 - *_all_spots.tif: The wavelet transform multiscale product segmentation of the original image.
 - *_filtered_spots.tif: Spots segmentation image with spots that don't correspond to the `MIN_SIZE` and `MIN_AXIS_LENGTH`
 parameters filtered out.
 - *_spots_in_mask.tif: The filtered spots image with only the spots within the dendritic mask.
 - *_mask.tif: Image of the dendritic mask.
 
**If `WRITE_HIST` is True**: A histogram of the coupling probability by the distance is saved as well.
 
All of these outputs ares saved in the specified `OUTPUT_DIRECTORY`.

## Experimental 3D OME-TIFF workflow

The 3D workflow is separate from the original 2D pySODA scripts:

```bash
python run_SODA_3D.py
```

By default, this targets the Henninger Lab 260512 CTD/MED1 OME-TIFF stack, uses channel 0 as CTD and channel 2 as MED1, reads OME voxel sizes, and writes 3D spot tables, nearest-neighbor distances, shell probabilities, a summary workbook, and a max-projection QC image. Use `--crop z0:z1,y0:y1,x0:x1` for fast validation before running a full stack.

The 3D workflow requires the scientific stack from the lab radial-plot/cellpose environment, including `tifffile`, `torch`, `scikit-image`, `scipy`, `pandas`, `openpyxl`, and `matplotlib`. By default it uses a separable MPS-friendly LoG implementation (`--log-method separable`) with YX tiling to avoid a slow dense 3D kernel. Use `--log-method dense` only as a correctness reference on cropped test regions.
