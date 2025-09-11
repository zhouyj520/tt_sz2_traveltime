# tt_sz2_traveltime

Python scripts to evaluate **SZ2** compression on 3D/4D traveltime tables  
and reproduce localization error statistics.

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Install SZ2

Clone and build SZ2 from the official repository:

```bash
git clone https://github.com/szcompressor/SZ2.git
cd SZ2
mkdir build && cd build
cmake .. && make -j
```

After compilation, make sure the `sz` executable is in your PATH,  
or specify its path explicitly with `--sz /path/to/sz`.

---

## 3. Run examples

### (a) Synthetic single-event sweep

```bash
python sz2_localization_pipeline.py   --n 170 --dx-km 0.02 --mode BOTH   --A-list 1e-4,8e-4,6e-4,4e-4,2e-4,1e-3,8e-3,6e-3,4e-3,2e-3,1e-2,8e-2,6e-2,4e-2,2e-2,1e-1   --R-list 1e-4,8e-4,6e-4,4e-4,2e-4,1e-3,8e-3,6e-3,4e-3,2e-3,1e-2,8e-2,6e-2,4e-2,2e-2,1e-1   --outdir _out_syn --sz sz
```

Generates `_out_syn/sz2_sweep_metrics.csv` with compression ratios  
and localization error metrics.

---

### (b) Multi-event evaluation (ABS=1 ms)

```bash
python sz2_abs001_multievent.py   --n 170 --dx-km 0.02 --A 0.001 --n-events 30   --outdir _out_multi --sz sz
```

Generates per-event location CSV and spatial plots.

---

### (c) Real traveltime volume sweep

```bash
python sz2_pipeline_real.py   --mat examples/TTP_overthrust_20_single.mat   --var auto --mode BOTH   --A-list ... --R-list ...   --outdir _out_real --sz sz
```

Produces converted `.dat` files and a metrics CSV.

---

## 4. Outputs

Each run writes a CSV file with:

- Compression ratio (CR)
- MAE, RMSE, p99(|Δt|), max(|Δt|)
- Localization differences (Δx, Δt₀, ΔRMS)

These CSVs can be used directly to regenerate performance plots.
