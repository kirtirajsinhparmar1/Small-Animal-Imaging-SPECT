# Small Animal Imaging SPECT

This repository contains the local ALO SPECT pipeline, including scanner geometry generation, T8 PPDF generation, beam analysis, JI calculation, visualization, dataset list generation, projection, and MLEM reconstruction.

## Main entry point

Run the full pipeline with:

```bash
python run_pipeline.py --run-name my_run --layout-idxs 0,1
```

The orchestrator runs the stages in order:

1. `generate_mph_scanner_circularfov.py`
2. `arg_ppdf_t8.py`
3. `arg_extract_beam_masks.py`
4. `arg_extract_beam_properties.py`
5. `arg_analyze_extracted_properties.py`
6. `6_calc_ji.py`
7. `generate_visuals.py`
8. `generate_flist.py`
9. `projection_t8.py`
10. `mlem_torch_gpf_nonmpi.py`
11. `view_npz.py`

## Output layout

Each run is isolated under `runs/<run-name>/`:

```text
data/      raw simulation and analysis outputs
plots/     geometry, analysis, and reconstruction visuals
results/   CSV and JSON summaries
recon/     projection arrays, phantom copy, reconstruction NPZ
logs/      per-stage command logs
```

## Notes

- PPDF generation is parallelized across poses.
- T8 beam masks, beam properties, and ASCI are computed once per layout from the in-memory sum of the eight T8 PPDF pose files. This matches the Omer baseline contract and produces `beams_masks_configuration_000.hdf5`, `beams_properties_configuration_000.hdf5`, and `asci_histogram_000.hdf5` style outputs.
- The pipeline preserves the existing stage scripts and only coordinates them.
- Scientific/output-affecting defaults live in the individual stage scripts. The pipeline only forwards those values when explicitly provided.
- If `--phantom` is provided, the phantom is copied into the run-local `recon/` directory; otherwise `projection_t8.py` uses its own default.
