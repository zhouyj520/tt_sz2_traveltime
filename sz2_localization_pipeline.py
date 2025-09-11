# -*- coding: utf-8 -*-
"""
Usage example:
    python sz2_localization_pipeline.py \
      --n 170 --dx-km 0.02 \
      --mode BOTH --A-list 1e-4,8e-4,6e-4,4e-4,2e-4,1e-3,8e-3,6e-3,4e-3,2e-3,1e-2,8e-2,6e-2,4e-2,2e-2,1e-1 \
      --R-list 1e-4,8e-4,6e-4,4e-4,2e-4,1e-3,8e-3,6e-3,4e-3,2e-3,1e-2,8e-2,6e-2,4e-2,2e-2,1e-1 \
      --outdir ./_SZ2_DEMO --sz sz
"""

import argparse, os, json, csv, time, math, shutil, subprocess
from pathlib import Path
import numpy as np
import skfmm
from typing import List, Tuple

# ---------- Utility Functions ----------
def which(exe: str) -> str:
    p = shutil.which(exe)
    if p is None:
        raise FileNotFoundError(f"Executable not found: {exe} (please specify the full path with --sz)")
    return p

def log(msg: str):
    print(msg, flush=True)

def run_cmd(cmd: List[str]) -> Tuple[int,str,str,float]:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip(), time.time()-t0

# ---------- Grid / Model ----------
def build_grid(n: int, dx_km: float):
    x = np.arange(n, dtype=np.float32) * dx_km
    y = np.arange(n, dtype=np.float32) * dx_km
    z = np.arange(n, dtype=np.float32) * dx_km
    L = (n - 1) * dx_km  # Cube side length (km)
    return x, y, z, L

def parse_layers(n: int, dx_km: float, L: float,
                 z_breaks_km: List[float]|None,
                 vp_list: List[float]|None,
                 vs_list: List[float]|None):
    """
    Layer parameters: z_breaks_km length = N+1 (including top/bottom), vp/vs length = N.
    If not provided, defaults to 3 layers: 0-L/3, L/3-2L/3, 2L/3-L, with Vp=[3.2,4.5,5.8] and Vs=Vp/√3 (km/s).
    """
    if not z_breaks_km or not vp_list or not vs_list:
        z_breaks_km = [0.0, L/3, 2*L/3, L]
        vp_list = [3.2, 4.5, 5.8]
        vs_list = [v/np.sqrt(3.0) for v in vp_list]
    assert len(z_breaks_km) == len(vp_list)+1 == len(vs_list)+1, "Mismatched layer parameter lengths"
    z_axis = np.arange(n, dtype=np.float32) * dx_km
    Vp = np.zeros((n,n,n), np.float32)
    Vs = np.zeros((n,n,n), np.float32)
    for i in range(len(vp_list)):
        z0, z1 = z_breaks_km[i], z_breaks_km[i+1]
        mask = (z_axis >= z0) & (z_axis < z1 if i < len(vp_list)-1 else z_axis <= z1)
        Vp[mask,:,:] = vp_list[i]
        Vs[mask,:,:] = vs_list[i]
    return Vp, Vs

# ---------- FMM Traveltime ----------
def fmm_traveltime_sec(src_xyz_km: Tuple[float,float,float],
                       V_kms: np.ndarray, dx_km: float) -> np.ndarray:
    """
    3D first-arrival traveltime volume (seconds) from a 'point source' (station).
    Uses skfmm.travel_time, with velocity field in km/s and grid step in km.
    """
    nz, ny, nx = V_kms.shape
    z = np.arange(nz, dtype=np.float32) * dx_km
    y = np.arange(ny, dtype=np.float32) * dx_km
    x = np.arange(nx, dtype=np.float32) * dx_km
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    sx, sy, sz = map(float, src_xyz_km)
    r0 = dx_km * 1.5
    # Eikonal level set: |∇T| = 1/V, initial narrow band phi=0 is a 'point sphere'
    phi = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2) - r0
    T_sec = skfmm.travel_time(phi, V_kms.astype(np.float32, copy=False), dx=dx_km)
    return np.maximum(T_sec, 0.0).astype(np.float32)

# ---------- Station Layout ----------
def make_four_corners(L: float, margin: float = 0.12):
    m = margin * L
    return [
        dict(name="SURF01", x_km=m,     y_km=m,     z_km=0.0),
        dict(name="SURF02", x_km=L-m,   y_km=m,     z_km=0.0),
        dict(name="SURF03", x_km=L-m,   y_km=L-m,   z_km=0.0),
        dict(name="SURF04", x_km=m,     y_km=L-m,   z_km=0.0),
    ]

# ---------- Trilinear Sampling ----------
def trilinear(vol: np.ndarray, x_km: float, y_km: float, z_km: float, dx_km: float) -> float:
    nz, ny, nx = vol.shape
    fx = np.clip(x_km / dx_km, 0.0, nx - 1 - 1e-7)
    fy = np.clip(y_km / dx_km, 0.0, ny - 1 - 1e-7)
    fz = np.clip(z_km / dx_km, 0.0, nz - 1 - 1e-7)
    x0 = int(np.floor(fx)); y0 = int(np.floor(fy)); z0 = int(np.floor(fz))
    tx = fx - x0; ty = fy - y0; tz = fz - z0
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
    c000 = vol[z0,y0,x0]; c100 = vol[z0,y0,x1]; c010 = vol[z0,y1,x0]; c110 = vol[z0,y1,x1]
    c001 = vol[z1,y0,x0]; c101 = vol[z1,y0,x1]; c011 = vol[z1,y1,x0]; c111 = vol[z1,y1,x1]
    c00 = c000*(1-tx) + c100*tx; c01 = c001*(1-tx) + c101*tx
    c10 = c010*(1-tx) + c110*tx; c11 = c011*(1-tx) + c111*tx
    c0  = c00*(1-ty) + c10*ty;  c1  = c01*(1-ty) + c11*ty
    return float(c0*(1-tz) + c1*tz)

# ---------- Continuous Localization (4-parameter: x,y,z,t0) ----------
def locate_xyzt0(tt_stack: np.ndarray, dx_km: float, t_obs: np.ndarray, L: float,
                 max_iter: int = 60, tol_step: float = 1e-6, tol_cost: float = 1e-12):
    """
    Objective function: min sum_i [ t0 + T_i(x,y,z) - t_obs_i ]^2
    - tt_stack: (n_chan, nz, ny, nx) channel order is arbitrary but fixed.
    - t_obs: Observed arrival times (seconds) corresponding to each channel.
    Returns: x,y,z,t0,RMS (km, km, km, s, s)
    """
    x = y = z = L/2.0
    t0 = 0.0
    lam = 1e-3  # LM damping
    n = t_obs.size

    def predict_vec(x,y,z):
        return np.array([trilinear(tt_stack[i], x,y,z, dx_km) for i in range(tt_stack.shape[0])], dtype=np.float64)

    for _ in range(max_iter):
        T = predict_vec(x,y,z)
        r = (t0 + T) - t_obs
        cost0 = float(np.dot(r,r))

        # Numerical Jacobian (central difference), column order: [x,y,z,t0]
        eps = max(0.1*dx_km, 1e-4)  # km
        T_px = predict_vec(min(x+eps,L), y, z); T_mx = predict_vec(max(x-eps,0.0), y, z)
        T_py = predict_vec(x, min(y+eps,L), z); T_my = predict_vec(x, max(y-eps,0.0), z)
        T_pz = predict_vec(x, y, min(z+eps,L)); T_mz = predict_vec(x, y, max(z-eps,0.0))
        J = np.zeros((n,4), dtype=np.float64)
        J[:,0] = (T_px - T_mx)/(2*eps)  # dT/dx
        J[:,1] = (T_py - T_my)/(2*eps)  # dT/dy
        J[:,2] = (T_pz - T_mz)/(2*eps)  # dT/dz
        J[:,3] = 1.0                    # dt0/dt0
        H = J.T @ J + lam*np.eye(4)
        g = J.T @ r
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        x_new = float(np.clip(x + step[0], 0.0, L))
        y_new = float(np.clip(y + step[1], 0.0, L))
        z_new = float(np.clip(z + step[2], 0.0, L))
        t0_new= float(t0 + step[3])
        T_new = predict_vec(x_new,y_new,z_new)
        r_new = (t0_new + T_new) - t_obs
        cost1 = float(np.dot(r_new,r_new))
        if cost1 < cost0:
            x,y,z,t0 = x_new,y_new,z_new,t0_new
            lam = max(lam/3.0, 1e-6)
            if max(abs(step[0]),abs(step[1]),abs(step[2])) < tol_step or abs(cost0-cost1) < tol_cost:
                break
        else:
            lam *= 5.0

    rms = math.sqrt(float(np.mean(((t0 + predict_vec(x,y,z)) - t_obs)**2)))
    return x,y,z,t0,rms

# ---------- SZ2 Compression/Decompression ----------
def sz2_compress(dat_path: Path, shape4: Tuple[int,int,int,int], sz_exec: str,
                 mode: str, A: float|None, R: float|None, tag: str):
    n_chan, nz, ny, nx = shape4
    cmd = [sz_exec, "-z", "-f", "-M", mode, "-i", str(dat_path),
           "-4", str(n_chan), str(nz), str(ny), str(nx)]
    if A is not None: cmd += ["-A", f"{A:g}"]
    if R is not None: cmd += ["-R", f"{R:g}"]
    rc, out, err, dt = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"SZ compression failed: {err or out}")
    # Standard output name <input>.sz or <input>.dat.sz
    raw1 = Path(str(dat_path)+".sz"); raw2 = Path(str(dat_path)+".dat.sz")
    raw = raw1 if raw1.exists() else (raw2 if raw2.exists() else None)
    if raw is None:
        raise FileNotFoundError("SZ output (.sz/.dat.sz) not found")
    suffix = f"_M{mode}"
    if A is not None: suffix += f"_A{A:g}"
    if R is not None: suffix += f"_R{R:g}"
    if tag:          suffix += f"_{tag}"
    out_sz = dat_path.parent / f"{dat_path.stem}{suffix}.sz"
    if out_sz.exists(): out_sz.unlink()
    raw.rename(out_sz)
    before, after = dat_path.stat().st_size, out_sz.stat().st_size
    return out_sz, dt, before, after

def sz2_decompress(sz_path: Path, shape4: Tuple[int,int,int,int], sz_exec: str):
    n_chan, nz, ny, nx = shape4
    out_path = Path(str(sz_path)+".out")
    if out_path.exists(): out_path.unlink()
    cmd = [sz_exec, "-x", "-f", "-s", str(sz_path), "-4", str(n_chan), str(nz), str(ny), str(nx)]
    rc, out, err, dt = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"SZ decompression failed: {err or out}")
    return out_path, dt

# ---------- Evaluate one point (one ABS or one REL) ----------
def eval_one(T_stack: np.ndarray, dat_path: Path, dx_km: float, L: float,
             true_xyz: Tuple[float,float,float], t0_true: float,
             sz_exec: str, mode: str, A: float|None, R: float|None, unit: str):
    n_chan, nz, ny, nx = T_stack.shape
    # Compress
    tag = unit
    sz_path, t_comp, before, after = sz2_compress(dat_path, (n_chan,nz,ny,nx), sz_exec, mode, A, R, tag)
    ratio = (before/after) if after>0 else np.inf
    # Decompress
    out_path, t_decomp = sz2_decompress(sz_path, (n_chan,nz,ny,nx), sz_exec)
    T_dec = np.fromfile(out_path, dtype=np.float32).reshape(T_stack.shape)

    # Error (traveltime volume)
    diff = (T_dec - T_stack).ravel()
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    p99  = float(np.quantile(np.abs(diff), 0.99))
    mx   = float(np.max(np.abs(diff)))

    # Compliance check (ABS: maxabs<=A; REL: maxabs<=R*range)
    rng = float(T_stack.max() - T_stack.min())
    bound_ok, bound_desc = True, ""
    if mode == "ABS" and A is not None:
        bound_ok = (mx <= A + 1e-12)
        bound_desc = f"A={A:g} {unit}, max|Δt|={mx:.3e}"
    elif mode == "REL" and R is not None:
        bound_ok = (mx <= R*rng + 1e-12)
        bound_desc = f"R={R:g} (~A_eq={R*rng:.3e} {unit}), max|Δt|={mx:.3e}"

    # Continuous localization (using t_obs from uncompressed volume as observation)
    def sample_vec(stack, x,y,z):
        return np.array([trilinear(stack[i], x,y,z, dx_km) for i in range(n_chan)], dtype=np.float64)
    t_obs = t0_true + sample_vec(T_stack, *true_xyz)
    x0_o,y0_o,z0_o,t00_o,rms_o = locate_xyzt0(T_stack, dx_km, t_obs, L)  # Reference solution
    x0_c,y0_c,z0_c,t00_c,rms_c = locate_xyzt0(T_dec,  dx_km, t_obs, L)  # Compressed solution

    dloc_km = math.sqrt((x0_c-x0_o)**2 + (y0_c-y0_o)**2 + (z0_c-z0_o)**2)
    dt0_s   = float(t00_c - t00_o)
    dRMS_s  = float(rms_c - rms_o)

    rec = dict(
        mode=mode, A=A, R=R, unit=unit,
        A_equiv=(R*rng if (mode=="REL" and R is not None) else None),
        R_equiv=(A/rng if (mode=="ABS" and A is not None) else None),
        size_in_MB=before/1e6, size_out_MB=after/1e6, ratio=ratio,
        t_comp_s=t_comp, t_decomp_s=t_decomp,
        mae=mae, rmse=rmse, p99=p99, maxabs=mx,
        loc_ref_x_km=x0_o, loc_ref_y_km=y0_o, loc_ref_z_km=z0_o, t0_ref_s=t00_o, rms_ref_s=rms_o,
        loc_cmp_x_km=x0_c, loc_cmp_y_km=y0_c, loc_cmp_z_km=z0_c, t0_cmp_s=t00_c, rms_cmp_s=rms_c,
        dloc_km=dloc_km, dt0_s=dt0_s, dRMS_s=dRMS_s,
        bound_ok=bool(bound_ok), bound_desc=bound_desc,
        sz_path=str(sz_path), out_path=str(out_path)
    )
    return rec

# ---------- Metadata ----------
def save_meta(dat_path: Path, shape4, dx_km: float, phases: List[str], stations: List[str]):
    meta = dict(
        dtype="float32",
        dims=list(shape4),
        dx_km=float(dx_km),
        phases=phases,
        stations=stations,
        description="Traveltime stack [n_chan, nz, ny, nx]"
    )
    with open(dat_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[META] {dat_path.with_suffix('.json')}")

# ---------- Main Process ----------
def main():
    ap = argparse.ArgumentParser(description="3D traveltime SZ2 compression-localization validation (ABS/REL sweep, simplified Save-All version)")
    ap.add_argument("--n", type=int, default=170, help="Number of grid points per axis")
    ap.add_argument("--dx-km", type=float, default=0.02, help="Grid spacing (km)")
    ap.add_argument("--outdir", default="./_SZ2_RUN", help="Output directory")
    ap.add_argument("--sz", default="sz", help="Path to SZ2 executable (or name if in PATH)")
    ap.add_argument("--mode", choices=["ABS","REL","BOTH"], default="BOTH", help="Scanning mode")
    # Threshold sweep
    ap.add_argument("--A-list", type=str, default="1e-3,3e-3,1e-2,3e-2,1e-1", help="ABS threshold list (seconds, comma-separated)")
    ap.add_argument("--R-list", type=str, default="0.001,0.003,0.01,0.03,0.05", help="REL threshold list (unitless)")
    # Layer model parameters (optional)
    ap.add_argument("--z-breaks-km", type=str, default="", help="Layer interface depths (km, comma-separated, length=N+1)")
    ap.add_argument("--vp-kms", type=str, default="", help="Vp for each layer (km/s, length=N)")
    ap.add_argument("--vs-kms", type=str, default="", help="Vs for each layer (km/s, length=N)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sz_exec = which(args.sz)

    # Grid and layer model
    x,y,z,L = build_grid(args.n, args.dx_km)
    if args.z_breaks_km and args.vp_kms and args.vs_kms:
        z_breaks = [float(s) for s in args.z_breaks_km.replace("，",",").split(",")]
        vp_list  = [float(s) for s in args.vp_kms.replace("，",",").split(",")]
        vs_list  = [float(s) for s in args.vs_kms.replace("，",",").split(",")]
    else:
        z_breaks = []; vp_list=[]; vs_list=[]
    Vp, Vs = parse_layers(args.n, args.dx_km, L, z_breaks, vp_list, vs_list)
    log(f"[GRID] n={args.n}, dx={args.dx_km} km, L={L:.3f} km")
    log(f"[LAYER] layers={len(vp_list) if vp_list else 3} (default=3 layers)")

    # Stations and phase channels (4 stations × 2 phases = 8 channels)
    stas = make_four_corners(L)
    phases = ["P","S"]
    chan_names = [f"{s['name']}-{ph}" for s in stas for ph in phases]
    n_chan = len(chan_names)

    # Generate traveltime volume for each channel
    T_list = []
    log("[FMM] Calculating P/S first-arrival traveltime for each station (seconds) ...")
    for s in stas:
        Tp = fmm_traveltime_sec((s["x_km"], s["y_km"], s["z_km"]), Vp, args.dx_km)
        Ts = fmm_traveltime_sec((s["x_km"], s["y_km"], s["z_km"]), Vs, args.dx_km)
        T_list += [Tp, Ts]
    T_stack = np.stack(T_list, axis=0).astype(np.float32)  # [n_chan, nz, ny, nx]

    # Write original volume (for SZ2 use)
    dat_path = outdir / f"tt_n{args.n}_dx{args.dx_km:g}_s_4d_f32.dat"
    T_stack.tofile(dat_path)
    save_meta(dat_path, T_stack.shape, args.dx_km, phases, [s['name'] for s in stas])

    # "True event" and observed arrival times (trilinear interpolation from uncompressed volume)
    true_xyz = (L/2, L/2, L/2)
    t0_true  = 0.0
    def sample_vec(stack, x,y,z):
        return np.array([trilinear(stack[i], x,y,z, args.dx_km) for i in range(n_chan)], dtype=np.float64)
    t_obs = t0_true + sample_vec(T_stack, *true_xyz)

    # Baseline localization (uncompressed volume)
    x_ref,y_ref,z_ref,t0_ref,rms_ref = locate_xyzt0(T_stack, args.dx_km, t_obs, L)
    log(f"[LOC-REF] true=({true_xyz[0]:.3f},{true_xyz[1]:.3f},{true_xyz[2]:.3f}) km | "
        f"ref=({x_ref:.3f},{y_ref:.3f},{z_ref:.3f}) km, t0_ref={t0_ref:.6f} s, RMS={rms_ref:.3e} s")

    # Sweep lists
    A_list = [float(s) for s in args.A_list.replace("，",",").split(",")] if args.mode in ("ABS","BOTH") else []
    R_list = [float(s) for s in args.R_list.replace("，",",").split(",")] if args.mode in ("REL","BOTH") else []
    scan = []
    for a in A_list: scan.append(("ABS", a, None))
    for r in R_list: scan.append(("REL", None, r))

    # Write summary CSV
    csv_path = outdir / "sz2_sweep_metrics.csv"
    fields = ["mode","A","R","unit","A_equiv","R_equiv","size_in_MB","size_out_MB","ratio",
              "t_comp_s","t_decomp_s","mae","rmse","p99","maxabs",
              "loc_ref_x_km","loc_ref_y_km","loc_ref_z_km","t0_ref_s","rms_ref_s",
              "loc_cmp_x_km","loc_cmp_y_km","loc_cmp_z_km","t0_cmp_s","rms_cmp_s",
              "dloc_km","dt0_s","dRMS_s","bound_ok","bound_desc","sz_path","out_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        for mode, A, R in scan:
            log(f"\n[SCAN] {mode} = {A if mode=='ABS' else R}")
            rec = eval_one(T_stack, dat_path, args.dx_km, L, true_xyz, t0_true, sz_exec,
                           mode=mode, A=A, R=R, unit="s")
            w.writerow(rec)
            log(f"[COMP] {rec['size_in_MB']:.2f} MB -> {rec['size_out_MB']:.2f} MB | CR={rec['ratio']:.2f}x")
            if mode=="ABS":
                log(f"[ERR ] mae={rec['mae']:.3e}, rmse={rec['rmse']:.3e}, p99={rec['p99']:.3e}, max={rec['maxabs']:.3e} s | R_eq={rec['R_equiv']:.3e}")
            else:
                log(f"[ERR ] mae={rec['mae']:.3e}, rmse={rec['rmse']:.3e}, p99={rec['p99']:.3e}, max={rec['maxabs']:.3e} s | A_eq={rec['A_equiv']:.3e} s")
            log(f"[LOC ] dloc={rec['dloc_km']*1000:.2f} m, dt0={rec['dt0_s']:.3e} s, dRMS={rec['dRMS_s']:.3e} s | bound_ok={rec['bound_ok']}")

    log(f"\n[OK] Sweep completed -> {csv_path}")
    log("[OK] All .sz and .sz.out files have been kept (distinguished by threshold and mode)")

if __name__ == "__main__":
    main()