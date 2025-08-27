# tt_sz2_traveltime

Error-bounded compression of 3-D traveltime tables (ABS/REL) with SZ2, plus MATLAB scripts to reproduce Figure 8 of the manuscript.

**Highlights**
- Reproducible ABS/REL compression on a real 3-D traveltime table.
- Python drivers call SZ2; MATLAB scripts generate Figure 8 slice comparisons.
- Millisecond-level accuracy preserved at ≈3–7× size reduction (see paper).

## 1. Data
`data/TTP_overthrust_20_single.mat` — real traveltime table used in our tests (grid spacing 20 m).

 If this file is large, Git LFS is recommended. See `.gitattributes`.

## 2. Dependencies
- **SZ2** (upstream): https://github.com/szcompressor/SZ2  
  Build SZ2 (CMake ≥3.16; GCC/Clang/MSVC). Make sure the CLI or shared lib is in PATH/LD_LIBRARY_PATH.
- Python ≥3.9
- MATLAB R2019b+ (tested)

## 3. Quick start
```bash
# ABS compression (absolute error bound, e.g., 1e-5)
python src/python/compress_abs.py \
  --input data/TTP_overthrust_20_single.mat \
  --abs 1e-5 \
  --out results/abs

# REL compression (relative error bound, e.g., 1e-5)
python src/python/compress_rel.py \
  --input data/TTP_overthrust_20_single.mat \
  --rel 1e-5 \
  --out results/rel
