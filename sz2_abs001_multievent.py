# -*- coding: utf-8 -*-
"""
Example:
    python sz2_abs001_multievent.py \
      --n 170 --dx-km 0.02 --A 0.001 --n-events 30 \
      --outdir _paper_multi --sz sz
"""

import argparse, csv, json, math, shutil, subprocess, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import skfmm
# 如果你也要出图，可再加：
# import matplotlib.pyplot as plt


# ------------------ Utilities ------------------
def which(exe: str) -> str:
    p = shutil.which(exe)
    if p is None:
        raise FileNotFoundError(
            f"Executable not found: {exe} (please specify the full path with --sz)")
    return p


def log(msg: str):
    print(msg, flush=True)


def run_cmd(cmd: List[str]):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip(), time.time() - t0


# ------------------ Grid & Layers ------------------
def build_grid(n: int, dx_km: float):
    x = np.arange(n, dtype=np.float32) * dx_km
    y = np.arange(n, dtype=np.float32) * dx_km
    z = np.arange(n, dtype=np.float32) * dx_km
    L = (n - 1) * dx_km
    return x, y, z, L


def parse_layers(n: int, dx_km: float, L: float,
                 z_breaks_km: Optional[List[float]] = None,
                 vp_list: Optional[List[float]] = None,
                 vs_list: Optional[List[float]] = None):
    """Return (Vp, Vs) both shaped [nz, ny, nx] in km/s."""
    if not z_breaks_km or not vp_list or not vs_list:
        z_breaks_km = [0.0, L / 3, 2 * L / 3, L]
        vp_list = [3.2, 4.5, 5.8]
        vs_list = [v / np.sqrt(3.0) for v in vp_list]
    assert len(z_breaks_km) == len(vp_list) + 1 == len(vs_list) + 1, \
        "Layer parameter lengths do not match"

    z_axis = np.arange(n, dtype=np.float32) * dx_km
    Vp = np.zeros((n, n, n), np.float32)
    Vs = np.zeros((n, n, n), np.float32)
    for i in range(len(vp_list)):
        z0, z1 = z_breaks_km[i], z_breaks_km[i + 1]
        mask = (z_axis >= z0) & (z_axis < z1 if i < len(vp_list) - 1 else z_axis <= z1)
        Vp[mask, :, :] = vp_list[i]
        Vs[mask, :, :] = vs_list[i]
    return Vp, Vs


# ------------------ Traveltime (FMM) ------------------
def fmm_traveltime_sec(src_xyz_km: Tuple[float, float, float],
                       V_kms: np.ndarray, dx_km: float) -> np.ndarray:
    """First-arrival traveltime from a point source using skfmm.travel_time."""
    nz, ny, nx = V_kms.shape
    z = np.arange(nz, dtype=np.float32) * dx_km
    y = np.arange(ny, dtype=np.float32) * dx_km
    x = np.arange(nx, dtype=np.float32) * dx_km
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    sx, sy, sz = map(float, src_xyz_km)
    r0 = dx_km * 1.5  # small sphere as initial front
    phi = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + (Z - sz) ** 2) - r0  # signed distance
    T_sec = skfmm.travel_time(phi, V_kms.astype(np.float32, copy=False), dx=dx_km)
    return np.maximum(T_sec, 0.0).astype(np.float32)


# ------------------ Stations & Sampling ------------------
def make_four_corners(L: float, margin: float = 0.12):
    m = margin * L
    return [
        dict(name="SURF01", x_km=m,   y_km=m,   z_km=0.0),
        dict(name="SURF02", x_km=L-m, y_km=m,   z_km=0.0),
        dict(name="SURF03", x_km=L-m, y_km=L-m, z_km=0.0),
        dict(name="SURF04", x_km=m,   y_km=L-m, z_km=0.0),
    ]


def trilinear(vol: np.ndarray, x_km: float, y_km: float, z_km: float, dx_km: float) -> float:
    nz, ny, nx = vol.shape
    fx = np.clip(x_km / dx_km, 0.0, nx - 1 - 1e-7)
    fy = np.clip(y_km / dx_km, 0.0, ny - 1 - 1e-7)
    fz = np.clip(z_km / dx_km, 0.0, nz - 1 - 1e-7)
    x0 = int(np.floor(fx)); y0 = int(np.floor(fy)); z0 = int(np.floor(fz))
    tx = fx - x0; ty = fy - y0; tz = fz - z0
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
    c000 = vol[z0, y0, x0]; c100 = vol[z0, y0, x1]; c010 = vol[z0, y1, x0]; c110 = vol[z0, y1, x1]
    c001 = vol[z1, y0, x0]; c101 = vol[z1, y0, x1]; c011 = vol[z1, y1, x0]; c111 = vol[z1, y1, x1]
    c00 = c000 * (1 - tx) + c100 * tx; c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx; c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty;   c1 = c01 * (1 - ty) + c11 * ty
    return float(c0 * (1 - tz) + c1 * tz)


# ------------------ Continuous localization (LM) ------------------
def locate_xyzt0(tt_stack: np.ndarray, dx_km: float, t_obs: np.ndarray, L: float,
                 max_iter: int = 60, tol_step: float = 1e-6, tol_cost: float = 1e-12):
    # init at box center
    x = y = z = L / 2.0
    t0 = 0.0
    lam = 1e-3
    n = t_obs.size

    def predict_vec(x, y, z):
        return np.array([trilinear(tt_stack[i], x, y, z, dx_km)
                         for i in range(tt_stack.shape[0])], dtype=np.float64)

    for _ in range(max_iter):
        T = predict_vec(x, y, z)
        r = (t0 + T) - t_obs
        cost0 = float(np.dot(r, r))

        eps = max(0.1 * dx_km, 1e-4)  # km
        T_px = predict_vec(min(x + eps, L), y, z); T_mx = predict_vec(max(x - eps, 0.0), y, z)
        T_py = predict_vec(x, min(y + eps, L), z); T_my = predict_vec(x, max(y - eps, 0.0), z)
        T_pz = predict_vec(x, y, min(z + eps, L)); T_mz = predict_vec(x, y, max(z - eps, 0.0))

        J = np.zeros((n, 4), dtype=np.float64)
        J[:, 0] = (T_px - T_mx) / (2 * eps)
        J[:, 1] = (T_py - T_my) / (2 * eps)
        J[:, 2] = (T_pz - T_mz) / (2 * eps)
        J[:, 3] = 1.0

        H = J.T @ J + lam * np.eye(4)
        g = J.T @ r
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        x_new = float(np.clip(x + step[0], 0.0, L))
        y_new = float(np.clip(y + step[1], 0.0, L))
        z_new = float(np.clip(z + step[2], 0.0, L))
        t0_new = float(t0 + step[3])

        T_new = predict_vec(x_new, y_new, z_new)
        r_new = (t0_new + T_new) - t_obs
        cost1 = float(np.dot(r_new, r_new))
        if cost1 < cost0:
            x, y, z, t0 = x_new, y_new, z_new, t0_new
            lam = max(lam / 3.0, 1e-6)
            if max(abs(step[0]), abs(step[1]), abs(step[2])) < tol_step or abs(cost0 - cost1) < tol_cost:
                break
        else:
            lam *= 5.0

    rms = math.sqrt(float(np.mean(((t0 + predict_vec(x, y, z)) - t_obs) ** 2)))
    return x, y, z, t0, rms


# ------------------ SZ2 I/O ------------------
def sz2_compress(dat_path: Path, shape4: Tuple[int, int, int, int], sz_exec: str, A: float):
    n_chan, nz, ny, nx = shape4
    cmd = [sz_exec, "-z", "-f", "-M", "ABS", "-i", str(dat_path),
           "-4", str(n_chan), str(nz), str(ny), str(nx), "-A", f"{A:g}"]
    rc, out, err, dt = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"SZ compression failed: {err or out}")
    raw1 = Path(str(dat_path) + ".sz")
    raw2 = Path(str(dat_path) + ".dat.sz")
    raw = raw1 if raw1.exists() else (raw2 if raw2.exists() else None)
    if raw is None:
        raise FileNotFoundError("SZ output (.sz/.dat.sz) not found")
    out_sz = dat_path.parent / f"{dat_path.stem}_MABS_A{A:g}.sz"
    if out_sz.exists():
        out_sz.unlink()
    raw.rename(out_sz)
    before, after = dat_path.stat().st_size, out_sz.stat().st_size
    return out_sz, dt, before, after


def sz2_decompress(sz_path: Path, shape4: Tuple[int, int, int, int], sz_exec: str):
    n_chan, nz, ny, nx = shape4
    out_path = Path(str(sz_path) + ".out")
    if out_path.exists():
        out_path.unlink()
    cmd = [sz_exec, "-x", "-f", "-s", str(sz_path),
           "-4", str(n_chan), str(nz), str(ny), str(nx)]
    rc, out, err, dt = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"SZ decompression failed: {err or out}")

    expected = n_chan * nz * ny * nx
    arr = np.fromfile(out_path, dtype=np.float32)
    if arr.size != expected:
        raise RuntimeError(f"Decompressed size mismatch: read {arr.size}, expected {expected}")
    return out_path, dt


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser(
        description="SZ2 Multi-event (ABS=0.001) evaluation and spatial statistics")
    ap.add_argument("--n", type=int, default=170)
    ap.add_argument("--dx-km", type=float, default=0.02)
    ap.add_argument("--outdir", default="./_SZ2_MULTI_ABS001")
    ap.add_argument("--sz", default="sz")
    ap.add_argument("--A", type=float, default=0.001, help="ABS threshold (seconds), default 0.001")
    ap.add_argument("--n-events", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--events-csv", type=str, default="",
                    help="If provided, reads events from CSV (cols: event_id,x_km,y_km,z_km,t0_s)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sz_exec = which(args.sz)

    # Grid & model
    x, y, z, L = build_grid(args.n, args.dx_km)
    Vp, Vs = parse_layers(args.n, args.dx_km, L)
    log(f"[GRID] n={args.n}, dx={args.dx_km} km, L={L:.3f} km")
    log(f"[LAYER] Three-layer velocity model, Vp(z)= [3.2, 4.5, 5.8] km/s")

    # Stations: four corners; channels: P,S per station => 8
    stas = make_four_corners(L)
    phases = ["P", "S"]

    # Traveltime stack [8, nz, ny, nx]
    log("[FMM] Computing P/S first arrival traveltimes for each station ...")
    T_list = []
    for s in stas:
        Tp = fmm_traveltime_sec((s["x_km"], s["y_km"], s["z_km"]), Vp, args.dx_km)
        Ts = fmm_traveltime_sec((s["x_km"], s["y_km"], s["z_km"]), Vs, args.dx_km)
        T_list += [Tp, Ts]
    T_stack = np.stack(T_list, axis=0).astype(np.float32)  # [8, nz, ny, nx]

    # Write original volume for SZ
    dat_path = outdir / f"tt_n{args.n}_dx{args.dx_km:g}_s_4d_f32.dat"
    T_stack.tofile(dat_path)
    meta = dict(
        dtype="float32",
        dims=[T_stack.shape[0], T_stack.shape[1], T_stack.shape[2], T_stack.shape[3]],
        dims_for_sz=[T_stack.shape[0], T_stack.shape[1], T_stack.shape[2], T_stack.shape[3]],
        dx_km=float(args.dx_km),
        extent_km=[L, L, L],
        phases=phases,
        stations=[s["name"] for s in stas],
    )
    (dat_path.with_suffix(".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[META] {dat_path.with_suffix('.json')}")

    # Compress & decompress (ABS)
    log(f"[SCAN] ABS = {args.A:g}")
    sz_path, t_comp, before, after = sz2_compress(dat_path, T_stack.shape, sz_exec, args.A)
    ratio = (before / after) if after > 0 else float("inf")
    out_path, t_decomp = sz2_decompress(sz_path, T_stack.shape, sz_exec)
    T_dec = np.fromfile(out_path, dtype=np.float32).reshape(T_stack.shape)
    log(f"[COMP] {before/1e6:.2f} MB -> {after/1e6:.2f} MB | CR={ratio:.2f}x")

    # Events
    if args.events_csv:
        events = []
        with open(args.events_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                events.append(dict(
                    event_id=row.get("event_id", ""),
                    x_km=float(row["x_km"]),
                    y_km=float(row["y_km"]),
                    z_km=float(row["z_km"]),
                    t0_s=float(row.get("t0_s", 0.0) or 0.0),
                ))
        if not events:
            raise ValueError(f"Event CSV is empty: {args.events_csv}")
        log(f"[EVENTS] Read {len(events)} events from CSV")
    else:
        rng = np.random.default_rng(args.seed)
        m = 0.02 * L  # keep away from boundaries a little
        xs = rng.uniform(m, L - m, size=args.n_events)
        ys = rng.uniform(m, L - m, size=args.n_events)
        zs = rng.uniform(m, L - m, size=args.n_events)
        events = [dict(event_id=f"ev{i+1:04d}",
                       x_km=float(xs[i]), y_km=float(ys[i]), z_km=float(zs[i]), t0_s=0.0)
                  for i in range(args.n_events)]
        log(f"[EVENTS] Randomly generated {len(events)} events (seed={args.seed})")

    # Localization (ref vs dec)
    def sample_vec(stack, x, y, z):
        return np.array([trilinear(stack[i], x, y, z, args.dx_km)
                         for i in range(T_stack.shape[0])], dtype=np.float64)

    per_event: List[Dict[str, float]] = []
    for ev in events:
        ex, ey, ez = ev["x_km"], ev["y_km"], ev["z_km"]
        t_obs = ev["t0_s"] + sample_vec(T_stack, ex, ey, ez)

        xr, yr, zr, t0r, rmsr = locate_xyzt0(T_stack, args.dx_km, t_obs, L)
        xc, yc, zc, t0c, rmsc = locate_xyzt0(T_dec,  args.dx_km, t_obs, L)

        per_event.append(dict(
            event_id=ev["event_id"],
            x_true_km=ex, y_true_km=ey, z_true_km=ez,
            x_ref_km=xr, y_ref_km=yr, z_ref_km=zr, t0_ref_s=t0r, rms_ref_s=rmsr,
            x_dec_km=xc, y_dec_km=yc, z_dec_km=zc, t0_dec_s=t0c, rms_dec_s=rmsc,
            dloc_km=float(math.sqrt((xc - xr) ** 2 + (yc - yr) ** 2 + (zc - zr) ** 2)),
            dt0_s=float(t0c - t0r), dRMS_s=float(rmsc - rmsr)
        ))

    # Export CSV (with a small schema guard)
    csv_path = outdir / "multievent_abs001_positions.csv"
    fields = ["event_id",
              "x_true_km", "y_true_km", "z_true_km",
              "x_ref_km", "y_ref_km", "z_ref_km", "t0_ref_s", "rms_ref_s",
              "x_dec_km", "y_dec_km", "z_dec_km", "t0_dec_s", "rms_dec_s",
              "dloc_km", "dt0_s", "dRMS_s"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in per_event:
            extra = set(r.keys()) - set(fields)
            missing = set(fields) - set(r.keys())
            if extra or missing:
                raise ValueError(f"CSV schema mismatch: extra={extra}, missing={missing}")
            w.writerow(r)
    log(f"[CSV] Per-event positions exported to -> {csv_path}")

    # Summary statistics
    dloc_m = np.array([r["dloc_km"] for r in per_event]) * 1000.0
    log("[STAT] d_loc (m): median={:.2f}, p25={:.2f}, p75={:.2f}, p95={:.2f}".format(
        np.median(dloc_m), np.percentile(dloc_m, 25),
        np.percentile(dloc_m, 75), np.percentile(dloc_m, 95)
    ))




if __name__ == "__main__":
    main()
