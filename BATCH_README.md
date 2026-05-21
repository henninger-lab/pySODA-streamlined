# Batch 2D pySODA Run Notes

This is the fast proof-of-concept workflow for folders of CZI, OME-TIFF, or analysis-ready multichannel TIFF images.
It prepares selected channels as 2D max projections and then runs pySODA.

## Input Format

The pipeline wrapper accepts:

- `.czi`
- `.ome.tif` / `.ome.tiff`
- regular `.tif` / `.tiff`

For the current CTD/MED1 proof of concept, select original channels:

- channel `0` = CTD
- channel `2` = MED1

The proof input currently lives here:

```bash
/Users/gwcmowry/Desktop/Henninger-Lab-Research/Code/Image-Analysis/pySODA/proof_of_concept_2D/input
```

## Simple Pipeline Command

This is the command to use for normal folder-scale runs. The selected channels are explicit CLI parameters; nothing biological is baked into the script.

```bash
cd /Users/gwcmowry/Desktop/Henninger-Lab-Research/Code/Image-Analysis/pySODA

MPLCONFIGDIR=/private/tmp/mpl \
/Users/gwcmowry/miniconda3/envs/cellpose-env/bin/python run_SODA_2D_pipeline.py \
  --input-dir "/path/to/full/input_images" \
  --output-dir "/path/to/pySODA_pipeline_output" \
  --channels 0 2
```

Default analysis settings:

- `--scale-list 3,4`
- `--scale-threshold 2`
- `--min-size 10`
- `--min-axis 3`
- `--min-intensity 0`
- `--roi-thresh 2.0`
- `--n-rings 20`
- `--ring-width 1`
- `--write-hist` is on by default

The numeric ring distance is in pixels. For these SIM images, `1 px = 0.040067807 um`, so a `--ring-width 1` ring is about `40.1 nm`.

To suppress histograms for faster large batches, add:

```bash
--no-write-hist
```

What this does:

- loads each `.czi`, `.tif`, `.tiff`, `.ome.tif`, or `.ome.tiff` in the input folder
- keeps only the requested channels, in the order listed by `--channels`
- max-projects Z if the input is a 3D stack
- writes analysis-ready `C,Y,X` TIFFs to `prepared_inputs`
- runs pySODA on the prepared selected-channel files
- writes `combined_pySODA_results.csv` and `combined_pySODA_results.xlsx`
- writes pooled cross-file outputs:
  - `pooled_pySODA_stats.csv`
  - `combined_SODA_rings.csv`
  - `pooled_SODA_rings.csv`
  - `pooled_SODA_rings_chX-chY.pdf`

For your CTD/MED1 case, the prepared pySODA channel `0` is original channel `0`/CTD, and prepared pySODA channel `1` is original channel `2`/MED1.

## Override Defaults

You only need these when tuning detection or SODA distance shells:

```bash
MPLCONFIGDIR=/private/tmp/mpl \
/Users/gwcmowry/miniconda3/envs/cellpose-env/bin/python run_SODA_2D_pipeline.py \
  --input-dir "/path/to/full/input_images" \
  --output-dir "/path/to/pySODA_pipeline_output" \
  --channels 0 2 \
  --scale-list 3,4 \
  --scale-threshold 2 \
  --min-size 10 \
  --min-axis 3 \
  --min-intensity 0 \
  --roi-thresh 2.0 \
  --n-rings 20 \
  --ring-width 1
```

If you select more than two channels, pySODA analyzes all selected channel pairs. Per-channel settings can be passed once for all selected channels or once per selected channel:

```bash
--channels 0 2 3 4 --min-size 10 10 20 20 --scale-list 3,4 3,4 4,5 4,5
```

## First-Image QC Command

For the first image or a tiny test folder, add ROI/spot QC outputs:

```bash
cd /Users/gwcmowry/Desktop/Henninger-Lab-Research/Code/Image-Analysis/pySODA

MPLCONFIGDIR=/private/tmp/mpl \
/Users/gwcmowry/miniconda3/envs/cellpose-env/bin/python run_SODA_2D_pipeline.py \
  --input-dir "/path/to/small_test_folder" \
  --output-dir "/path/to/pySODA_qc_output" \
  --channels 0 2 \
  --scale-list 3,4 \
  --scale-threshold 2 \
  --min-size 10 \
  --min-axis 3 \
  --min-intensity 0 \
  --roi-thresh 2.0 \
  --n-rings 20 \
  --ring-width 1 \
  --save-roi \
  --write-hist
```

Use this to inspect spot detection, masks, coupling-probability histograms, and directional coupling-percent plots before launching a large batch.

Note that the distance histogram is not truly directional: it is the pairwise coupling probability as a function of distance for the channel pair. The directionality comes from the coupling-index denominator, so pySODA now also writes a `coupling_percent_*_chXY.pdf` plot with both `chX -> chY` and `chY -> chX`.

## Outputs to Report

The dataset summary workbook is:

```bash
pySODA_results_[input-folder-name].xlsx
```

For each channel pair, report:

- spots in each channel
- number of coupled pairs
- coupling index for each direction
- coupling percent for each direction
- weighted mean coupling distance
- SODA global p-value display / `log10(p-value)`
- `G0 max`, `G0 threshold`, and significant ring ranges

Each per-image workbook also has:

- `SODA summary`: paper-style SODA metrics
- `SODA rings`: per-ring `G`, variance, `G0`, significance, and coupling probability
- `Couples`: coupled object pairs
- `Spots ch0`, `Spots ch1`: per-object spot tables

The pooled outputs summarize across files:

- `combined_pySODA_results.csv`: one row per file and channel pair
- `pooled_pySODA_stats.csv`: pooled spot counts, pooled coupling percentages, pooled weighted mean coupling distance, and per-file mean/SEM
- `combined_SODA_rings.csv`: one row per file, channel pair, and distance ring
- `pooled_SODA_rings.csv`: cross-file mean/SEM coupling probability per ring
- `pooled_SODA_rings_chX-chY.pdf`: pooled coupling-probability curve

Use pooled values as descriptive summaries. For condition-level statistics, use the per-file/per-cell rows in `combined_pySODA_results.csv` to avoid treating individual spots as independent biological replicates.

## Speed Tips

- Add `--no-write-hist` for the fastest large batches. The `SODA rings` sheet contains the same values for plotting later.
- Keep `--save-roi` off for batch runs after the detector has been validated.
- Include only the channels you want with `--channels`; this is safer than excluding channels when inputs have 4+ channels.
- If one selected/prepared channel should define the ROI mask, use `--channel-mask N` where `N` is the prepared channel index after selection.
- For paper figures, rerun a representative subset with `--save-roi --write-hist` to make QC material.
