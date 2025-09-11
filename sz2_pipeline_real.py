# -*- coding: utf-8 -*-
"""
Traveltime .mat -> .dat(4D) -> SZ2 compress/decompress -> error & compression ratio CSV
Example:
  python cs.py \
    --mat TTP_overthrust_20_single.mat \
    --var auto \
    --mode BOTH \
    --A-list 1e-4,8e-4,6e-4,4e-4,2e-4,1e-3,8e-3,6e-3,4e-3,2e-3,1e-2,8e-2,6e-2,4e-2,2e-2,1e-1 \
    --R-list 1e-4,8e-4,6e-4,4e-4,2e-4,1e-3,8e-3,6e-3,4e-3,2e-3,1e-2,8e-2,6e-2,4e-2,2e-2,1e-1 \
    --outdir ./_SZ2_TTP --sz sz
"""
import argparse, os, json, csv, time, shutil, subprocess, re
from pathlib import Path
import numpy as np

def log(m): print(m, flush=True)

def which(exe: str) -> str:
    p = shutil.which(exe)
    if p is None:
        raise FileNotFoundError(f"Executable not found: {exe} (Please use --sz to specify the full path)")
    return p

def run_cmd(cmd):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr, time.time()-t0

# ---------- Load .mat ----------
def load_mat_array(mat_path: Path, var: str|None):
    """
    Returns (arr, varname). arr is a np.ndarray with dtype=float32 and a 3D/4D shape.
    - If var='auto': Automatically selects the floating-point array with the most voxels.
    - Compatible with v7 (scipy.io.loadmat) and v7.3 (h5py).
    """
    mat_path = Path(mat_path)
    # Try h5py (v7.3)
    try:
        import h5py
        with h5py.File(mat_path, "r") as f:
            def all_dsets(h, prefix=""):
                out=[]
                for k,v in h.items():
                    if isinstance(v, h5py.Dataset):
                        out.append( (prefix+k, v) )
                    elif isinstance(v, h5py.Group):
                        out += all_dsets(v, prefix+k+"/")
                return out
            items = all_dsets(f, "")
            cand = []
            for name, ds in items:
                shp = tuple(ds.shape)
                if len(shp) in (3,4) and np.issubdtype(ds.dtype, np.floating):
                    cand.append((name, int(np.prod(shp))))
            if var and var!="auto":
                # Loose matching: /A/B/C or matching the last part of the name is acceptable
                def name_match(nm):
                    return (nm==var) or nm.split("/")[-1]==var
                choices = [nm for nm,_ in cand if name_match(nm)]
                if not choices: raise KeyError(f"HDF5 found no 3D/4D floating-point dataset named {var}")
                name = choices[0]
            else:
                if not cand: raise KeyError("No 3D/4D floating-point dataset found in HDF5 file")
                name = max(cand, key=lambda x:x[1])[0]
            arr = f[name][()]
            return np.asarray(arr, dtype=np.float32), name
    except Exception:
        pass

    # Fallback to scipy.io (v7)
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise RuntimeError("Reading v7 .mat requires scipy, or install h5py to read v7.3") from e
    md = loadmat(mat_path, squeeze_me=True, struct_as_string=False, chars_as_strings=True)
    # Filter internal keys
    keys = [k for k in md.keys() if not k.startswith("__")]
    cand=[]
    for k in keys:
        v = md[k]
        if isinstance(v, np.ndarray) and v.dtype.kind in ("f","d") and v.ndim in (3,4):
            cand.append((k, int(np.prod(v.shape))))
    if var and var!="auto":
        if var not in md or not (isinstance(md[var], np.ndarray) and md[var].ndim in (3,4)):
            raise KeyError(f"No 3D/4D floating-point array named {var} found in MAT file")
        arr = md[var].astype(np.float32, copy=False)
        return arr, var
    else:
        if not cand: raise KeyError("No 3D/4D floating-point array found in MAT file")
        name = max(cand, key=lambda x:x[1])[0]
        return md[name].astype(np.float32, copy=False), name

# ---------- Standardize shape to [n_chan, nz, ny, nx] ----------
def to_stack_4d(arr: np.ndarray, chan_axis: str|None):
    """
    Accepts 3D or 4D:
      - 3D: (nz,ny,nx) or (nx,ny,nz), etc. -> treated as single channel, with an attempt to intelligently identify the z-axis.
      - 4D: Common shapes are (nz,ny,nx,nc) or (nc,nz,ny,nx). Can specify --chan-axis as '0' or '-1'.
    Returns a stack with shape [n_chan, nz, ny, nx]
    """
    if arr.ndim==3:
        nz, ny, nx = arr.shape # Don't force reordering, as long as "write and read back are consistent", the error can be compared.
        stack = arr[None, ...] # n_chan=1
        return stack.astype(np.float32, copy=False)
    elif arr.ndim==4:
        if chan_axis in ("0","first"):
            stack = arr
        elif chan_axis in ("-1","last", None):
            # By default, assume the last dimension is the channel (many v7.3 exports are like this)
            if chan_axis in ("-1","last"):
                stack = np.moveaxis(arr, -1, 0)
            else:
                # Automatic detection: treat the "smallest dimension" as the channel
                ax = int(np.argmin(arr.shape))
                stack = np.moveaxis(arr, ax, 0)
        else:
            raise ValueError("--chan-axis only supports '0' or '-1'")
        if stack.shape[0] > 512 and min(stack.shape[1:]) < 8:
            # The channel axis is clearly misplaced, switch axes
            stack = np.moveaxis(stack, 0, -1)
            stack = np.moveaxis(stack, -1, 0)
        return stack.astype(np.float32, copy=False)
    else:
        raise ValueError("Only 3D/4D traveltime volumes are supported")

# ---------- SZ2 Compress/Decompress ----------
def sz2_compress(dat_path: Path, shape4, sz_exec: str, mode: str, A:float|None, R:float|None, tag="s"):
    n_chan, nz, ny, nx = shape4
    cmd = [sz_exec, "-z", "-f", "-M", mode, "-i", str(dat_path),
           "-4", str(n_chan), str(nz), str(ny), str(nx)]
    if A is not None: cmd += ["-A", f"{A:g}"]
    if R is not None: cmd += ["-R", f"{R:g}"]
    rc, out, err, dt = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"SZ compression failed: {err or out}")
    raw1 = Path(str(dat_path)+".sz"); raw2 = Path(str(dat_path)+".dat.sz")
    raw = raw1 if raw1.exists() else (raw2 if raw2.exists() else None)
    if raw is None: raise FileNotFoundError("SZ output (.sz/.dat.sz) not found")
    suffix = f"_M{mode}"
    if A is not None: suffix += f"_A{A:g}"
    if R is not None: suffix += f"_R{R:g}"
    if tag:          suffix += f"_{tag}"
    out_sz = dat_path.parent / f"{dat_path.stem}{suffix}.sz"
    if out_sz.exists(): out_sz.unlink()
    raw.rename(out_sz)
    before, after = dat_path.stat().st_size, out_sz.stat().st_size
    return out_sz, dt, before, after

def sz2_decompress(sz_path: Path, shape4, sz_exec: str):
    n_chan, nz, ny, nx = shape4
    out_path = Path(str(sz_path)+".out")
    if out_path.exists(): out_path.unlink()
    cmd = [sz_exec, "-x", "-f", "-s", str(sz_path), "-4", str(n_chan), str(nz), str(ny), str(nx)]
    rc, out, err, dt = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"SZ decompression failed: {err or out}")
    return out_path, dt

# ---------- Main Process ----------
def main():
    ap = argparse.ArgumentParser(description="Measured 3D/4D traveltime SZ2 compression sweep (generates CSV)")
    ap.add_argument("--mat", required=True, help=".mat file path")
    ap.add_argument("--var", default="auto", help="Variable name ('auto' to automatically select the 3D/4D float data with the most voxels)")
    ap.add_argument("--chan-axis", default=None, help="Position of the channel axis for multi-channel 4D data: '0' or '-1' (default: auto)")
    ap.add_argument("--outdir", default="./_SZ2_RUN", help="Output directory")
    ap.add_argument("--sz", default="sz", help="SZ2 executable path or command name")
    ap.add_argument("--mode", choices=["ABS","REL","BOTH"], default="BOTH", help="Sweep mode")
    ap.add_argument("--A-list", default="1e-5,3e-5,1e-4,3e-4,1e-3", help="ABS thresholds (seconds)")
    ap.add_argument("--R-list", default="1e-4,3e-4,1e-3,3e-3,1e-2", help="REL thresholds (dimensionless)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sz_exec = which(args.sz)

    # Read and reshape
    arr, picked = load_mat_array(Path(args.mat), args.var)
    log(f"[LOAD] {args.mat} | var='{picked}' | shape={arr.shape} | dtype={arr.dtype}")
    stack = to_stack_4d(arr, args.chan_axis) # [n_chan, nz, ny, nx]
    n_chan, nz, ny, nx = stack.shape
    log(f"[SHAPE] stack = [n_chan={n_chan}, nz={nz}, ny={ny}, nx={nx}]")

    # Write .dat + metadata
    dat_path = outdir / f"tt_real_{picked.replace('/','_')}_4d_f32.dat"
    stack.tofile(dat_path)
    meta = dict(dtype="float32", dims=[int(n_chan), int(nz), int(ny), int(nx)],
                description="Traveltime stack [n_chan, nz, ny, nx] from MAT")
    with open(dat_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[META] {dat_path.with_suffix('.json')}")

    # Sweep list
    A_list = [float(s) for s in re.split(r"[，,]", args.A_list.strip())] if args.mode in ("ABS","BOTH") else []
    R_list = [float(s) for s in re.split(r"[，,]", args.R_list.strip())] if args.mode in ("REL","BOTH") else []
    scan = []
    for a in A_list: scan.append(("ABS", a, None))
    for r in R_list: scan.append(("REL", None, r))

    # Calculate range (for REL -> A_equiv)
    rng = float(stack.max() - stack.min())

    # CSV
    csv_path = outdir / "sz2_sweep_metrics.csv"
    fields = ["mode","A","R","unit","A_equiv","R_equiv",
              "size_in_MB","size_out_MB","ratio","t_comp_s","t_decomp_s",
              "mae","rmse","p99","maxabs","bound_ok","bound_desc",
              "mat_path","var","dat_path","sz_path","out_path","shape"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()

        for mode, A, R in scan:
            log(f"\n[SCAN] {mode} = {A if A is not None else R}")
            # Compress
            sz_path, t_comp, before, after = sz2_compress(dat_path, stack.shape, sz_exec, mode, A, R, tag="s")
            ratio = (before/after) if after>0 else np.inf
            # Decompress
            out_path, t_decomp = sz2_decompress(sz_path, stack.shape, sz_exec)
            dec = np.fromfile(out_path, dtype=np.float32).reshape(stack.shape)

            # Error
            diff = (dec - stack).ravel()
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            p99 = float(np.quantile(np.abs(diff), 0.99))
            mx = float(np.max(np.abs(diff)))

            # Threshold compliance
            unit = "s"
            bound_ok, bdesc = True, ""
            if mode=="ABS":
                bound_ok = (mx <= A + 1e-12)
                bdesc = f"A={A:g} s, max|Δt|={mx:.3e}"
                Aeq, Req = None, (A/rng if rng>0 else None)
            else:
                bound_ok = (mx <= R*rng + 1e-12)
                bdesc = f"R={R:g} (~A_eq={R*rng:.3e} s), max|Δt|={mx:.3e}"
                Aeq, Req = (R*rng), None

            rec = dict(
                mode=mode, A=A, R=R, unit=unit, A_equiv=Aeq, R_equiv=Req,
                size_in_MB=before/1e6, size_out_MB=after/1e6, ratio=ratio,
                t_comp_s=t_comp, t_decomp_s=t_decomp,
                mae=mae, rmse=rmse, p99=p99, maxabs=mx,
                bound_ok=bool(bound_ok), bound_desc=bdesc,
                mat_path=str(Path(args.mat).resolve()), var=picked,
                dat_path=str(dat_path), sz_path=str(sz_path), out_path=str(out_path),
                shape=str(list(stack.shape))
            )
            w.writerow(rec)
            log(f"[COMP] {rec['size_in_MB']:.2f} MB -> {rec['size_out_MB']:.2f} MB | CR={rec['ratio']:.2f}x")
            log(f"[ERR ] mae={mae:.3e}, rmse={rmse:.3e}, p99={p99:.3e}, max={mx:.3e} s | bound_ok={bound_ok}")

    log(f"\n[OK] Sweep complete -> {csv_path}")
    log("[OK] .sz and .sz.out files have been kept for review/subsequent experiments")
if __name__ == "__main__":
    main()