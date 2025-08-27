#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存友好版（REL）：.mat(v7.3, HDF5) -> .dat(分块) -> SZ 压缩/解压(REL) -> 分块误差验证 -> CSV
与 ABS 版完全相同，唯一变化：压缩改为 REL（-M REL -R <bound>），命令行用 --rel。
用法示例：
python3 sz3d_benchmark_actual_REL.py \
  --mat20 "/home/yuanjian/三维走时数据/TTP_overthrust_20_single.mat" \
  --abs 1e-5 \
  --outdir "/home/yuanjian/三维走时数据/_SZ_RESULTS_REL"
"""

import os, json, math, time, csv, argparse, subprocess
from pathlib import Path
import numpy as np

try:
    from scipy.io import loadmat as _scipy_loadmat
except Exception:
    _scipy_loadmat = None

import h5py

# ---------------- 工具 ----------------
def human(n: int) -> str:
    u = ['B','KB','MB','GB','TB','PB']
    f = float(n)
    for uu in u:
        if f < 1024: return f"{f:.2f}{uu}"
        f /= 1024
    return f"{f:.2f}EB"

def run(cmd: list[str]) -> tuple[int,str,str,float]:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    return p.returncode, p.stdout.strip(), p.stderr.strip(), dt

def memmap_metrics(a_path: Path, b_path: Path, dtype_name: str, chunk_elems: int = 10_000_000):
    dt = np.float32 if dtype_name == 'float32' else np.float64
    a = np.memmap(a_path, mode='r', dtype=dt)
    b = np.memmap(b_path, mode='r', dtype=dt)
    if a.size != b.size:
        raise ValueError(f"长度不一致: {a.size} vs {b.size}")
    n = a.size
    gmax, sabs, ssq = 0.0, 0.0, 0.0
    dmin, dmax = None, None
    for st in range(0, n, chunk_elems):
        ed = min(n, st + chunk_elems)
        aa = np.array(a[st:ed], copy=False)
        bb = np.array(b[st:ed], copy=False)
        ee = np.abs(aa - bb)
        gmax = max(gmax, float(ee.max()))
        sabs += float(ee.sum()); ssq += float((ee*ee).sum())
        amin, amax = float(aa.min()), float(aa.max())
        dmin = amin if dmin is None else min(dmin, amin)
        dmax = amax if dmax is None else max(dmax, amax)
    mean_abs = sabs / n
    rmse = math.sqrt(ssq / n)
    peak = max(abs(dmin), abs(dmax)) or 1.0
    psnr = (10.0 * math.log10((peak*peak)/(rmse*rmse))) if rmse > 0 else float('inf')
    return gmax, mean_abs, rmse, psnr, peak

def estimate_step_from_shape(shape_xyz):
    nx, ny, nz = shape_xyz
    step_x = 16000.0 / (nx - 1) if nx > 1 else float('nan')
    step_y = 16000.0 / (ny - 1) if ny > 1 else float('nan')
    step_z = 3740.0  / (nz - 1) if nz > 1 else float('nan')
    return step_x, step_y, step_z

# ---------------- 读取/导出 .mat（分块写 .dat） ----------------
def _largest_3d_dataset_h5(f: h5py.File):
    best_name, best_size, best_ds = None, -1, None
    def visit(name, obj):
        nonlocal best_name, best_size, best_ds
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and np.issubdtype(obj.dtype, np.number):
            sz = obj.size
            if sz > best_size:
                best_name, best_size, best_ds = name, sz, obj
    f.visititems(visit)
    if best_ds is None:
        raise RuntimeError("未在 .mat(v7.3/HDF5) 中找到 3D 数值数组")
    return best_name, best_ds

def _largest_3d_array_scipy(md: dict):
    best_key, best_size, best_arr = None, -1, None
    for k, v in md.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and v.ndim == 3 and np.issubdtype(v.dtype, np.number):
            sz = v.size
            if sz > best_size:
                best_key, best_size, best_arr = k, sz, v
    if best_arr is None:
        raise RuntimeError("未在 .mat(<=7.2) 中找到 3D 数值数组")
    return best_key, best_arr

def export_mat_to_dat_chunked(mat_path: Path, dat_path: Path, meta_path: Path,
                              nominal_step_m: float, chunk_slices: int = 8):
    try:
        with h5py.File(mat_path, 'r') as f:
            key, ds = _largest_3d_dataset_h5(f)
            shape = tuple(int(x) for x in ds.shape)   # (nx, ny, nz)
            dtype_np = np.float32 if ds.dtype == np.float32 else np.float64
            dtype_flag = '-f' if dtype_np == np.float32 else '-d'
            dtype_name = 'float32' if dtype_np == np.float32 else 'float64'

            nx, ny, nz = shape
            dat_path.unlink(missing_ok=True)
            with open(dat_path, 'wb') as w:
                for k in range(0, nz, chunk_slices):
                    kz = min(nz, k + chunk_slices)
                    block = ds[:, :, k:kz]
                    blk = np.ascontiguousarray(block.astype(dtype_np, copy=False))
                    blk.tofile(w)

            step_est = estimate_step_from_shape(shape)
            meta = {
                "source_mat": str(mat_path),
                "var_or_name": key,
                "shape_for_sz": list(shape),
                "dtype": dtype_name,
                "dims_for_sz": [nx, ny, nz],
                "nominal_step_m": nominal_step_m,
                "estimated_node_step_m": list(step_est),
                "extent_m": [16000.0, 16000.0, 3740.0],
                "source_xyz_m": [8000.0, 8000.0, 2000.0],
                "note": "dat 为 C-order；SZ 使用 dims_for_sz；分块导出避免 OOM。"
            }
            with open(meta_path, "w", encoding="utf-8") as fmeta:
                json.dump(meta, fmeta, ensure_ascii=False, indent=2)

            return shape, dtype_name, dtype_flag

    except OSError:
        if _scipy_loadmat is None:
            raise
        md = _scipy_loadmat(str(mat_path), simplify_cells=True)
        key, arr = _largest_3d_array_scipy(md)
        shape = tuple(int(x) for x in arr.shape)
        dtype_np = np.float32 if (arr.dtype == np.float32 or "float32" in str(arr.dtype)) else np.float64
        dtype_flag = '-f' if dtype_np == np.float32 else '-d'
        dtype_name = 'float32' if dtype_np == np.float32 else 'float64'

        nx, ny, nz = shape
        dat_path.unlink(missing_ok=True)
        with open(dat_path, 'wb') as w:
            for k in range(nz):
                sl = np.ascontiguousarray(arr[:, :, k].astype(dtype_np, copy=False))
                sl.tofile(w)

        step_est = estimate_step_from_shape(shape)
        meta = {
            "source_mat": str(mat_path),
            "var_or_name": key,
            "shape_for_sz": list(shape),
            "dtype": dtype_name,
            "dims_for_sz": [nx, ny, nz],
            "nominal_step_m": nominal_step_m,
            "estimated_node_step_m": list(step_est),
            "extent_m": [16000.0, 16000.0, 3740.0],
            "source_xyz_m": [8000.0, 8000.0, 2000.0],
            "note": "dat 为 C-order；SZ 使用 dims_for_sz；旧版 .mat 回退。"
        }
        with open(meta_path, "w", encoding="utf-8") as fmeta:
            json.dump(meta, fmeta, ensure_ascii=False, indent=2)
        return shape, dtype_name, dtype_flag

# ---------------- 单数据集处理（REL） ----------------
def process_one(mat_path: Path, nominal_step_m: float, sz_exec: str, rel_err: float,
                force_rebuild_dat: bool, out_dir: Path, chunk_slices: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_path = out_dir / (mat_path.stem + ".dat")
    meta_path = out_dir / (mat_path.stem + ".json")

    # 若不存在 .dat 或要求重建，则分块导出
    if force_rebuild_dat or (not dat_path.exists()):
        print(f"[INFO] 分块导出 {mat_path.name} -> {dat_path.name} ...")
        shape, dtype_name, dtype_flag = export_mat_to_dat_chunked(
            mat_path, dat_path, meta_path, nominal_step_m, chunk_slices=chunk_slices
        )
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        shape = tuple(meta["dims_for_sz"])
        dtype_name = meta["dtype"]
        dtype_flag = '-f' if dtype_name == 'float32' else '-d'

    nx, ny, nz = shape
    elements = nx * ny * nz
    step_est = estimate_step_from_shape(shape)

    # 压缩（REL）
    comp_cmd = [sz_exec, "-z", dtype_flag, "-M", "REL", "-R", str(rel_err),
                "-i", str(dat_path), "-3", str(nx), str(ny), str(nz)]
    rc, out, err, t_comp = run(comp_cmd)
    if rc != 0:
        raise RuntimeError(f"SZ 压缩失败：{err or out}")
    # SZ 默认输出名仍是 .dat.sz
    sz_path = dat_path.with_suffix(".dat.sz")
    in_bytes  = dat_path.stat().st_size
    out_bytes = sz_path.stat().st_size
    ratio = (in_bytes / out_bytes) if out_bytes else float('inf')

    # 解压（输出 .dat.sz.out）
    decomp_cmd = [sz_exec, "-x", dtype_flag, "-s", str(sz_path), "-3", str(nx), str(ny), str(nz)]
    rc2, out2, err2, t_decomp = run(decomp_cmd)
    if rc2 != 0:
        raise RuntimeError(f"SZ 解压失败：{err2 or out2}")
    out_path = dat_path.with_suffix(".dat.sz.out")

    # 分块误差验证
    gmax, mean_abs, rmse, psnr, peak = memmap_metrics(dat_path, out_path, dtype_name, chunk_elems=10_000_000)
    max_ms, mean_ms, rmse_ms = gmax*1e3, mean_abs*1e3, rmse*1e3

    # 摘要
    print(f"\n=== {mat_path.name} | nominal step={nominal_step_m} m | dtype={dtype_name} | REL={rel_err} ===")
    print(f"shape (nodes) = {nx} x {ny} x {nz} | elements = {elements:,}")
    print(f"estimated node step ~ ({step_est[0]:.3f}, {step_est[1]:.3f}, {step_est[2]:.3f}) m")
    print(f"size: {human(in_bytes)} -> {human(out_bytes)} | ratio = {ratio:.2f}x")
    print(f"time: compress={t_comp:.3f}s, decompress={t_decomp:.3f}s")
    print(f"error: max={gmax:.3e}s ({max_ms:.3f} ms), mean={mean_abs:.3e}s ({mean_ms:.3f} ms), "
          f"RMSE={rmse:.3e}s ({rmse_ms:.3f} ms), PSNR={psnr:.2f} dB (peak={peak:.3g})")

    return {
        "dataset": mat_path.name,
        "nominal_step_m": nominal_step_m,
        "dtype": dtype_name,
        "nx": nx, "ny": ny, "nz": nz,
        "elements": elements,
        "rel_err_bound": rel_err,
        "orig_bytes": in_bytes,
        "comp_bytes": out_bytes,
        "ratio": ratio,
        "t_compress_s": t_comp,
        "t_decompress_s": t_decomp,
        "max_err_s": gmax,
        "mean_abs_err_s": mean_abs,
        "rmse_s": rmse,
        "max_err_ms": max_ms,
        "mean_abs_err_ms": mean_ms,
        "rmse_ms": rmse_ms,
        "psnr_db": psnr,
        "est_node_step_x_m": step_est[0],
        "est_node_step_y_m": step_est[1],
        "est_node_step_z_m": step_est[2],
    }

# ---------------- 主入口 ----------------
def main():
    ap = argparse.ArgumentParser(description="SZ2 实际三维走时体压缩/验证（REL 版，分块导出避免 OOM）")
    ap.add_argument("--sz", default="sz", help="sz 可执行（PATH 中）")
    ap.add_argument("--rel", type=float, default=1e-5, help="REL 相对误差上限（比例）")
    ap.add_argument("--force", action="store_true", help="若存在 .dat 仍重新导出")
    ap.add_argument("--outdir", default="./_sz_actual_out_rel", help="输出目录")
    ap.add_argument("--mat10", help="10 m 网格 .mat")
    ap.add_argument("--mat20", help="20 m 网格 .mat")
    ap.add_argument("--chunk-slices", type=int, default=8,
                    help="导出 .dat 时每次沿 Z 读取的切片层数（默认8，内存更紧张可调小，比如4/2）")
    ap.add_argument("--cast-f32", action="store_true", help="导出时强制转为 float32，减小体量（可选）")
    args = ap.parse_args()

    if not args.mat10 and not args.mat20:
        ap.error("必须至少提供 --mat10 或 --mat20 之一")

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    if args.mat10:
        recs.append(process_one(Path(args.mat10), 10.0, args.sz, args.rel, args.force, out_dir, chunk_slices=args.chunk_slices))
    if args.mat20:
        recs.append(process_one(Path(args.mat20), 20.0, args.sz, args.rel, args.force, out_dir, chunk_slices=args.chunk_slices))

    if recs:
        csv_path = out_dir / "summary_actual_tt_compression.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)
        print(f"\n[OK] 已生成对比表：{csv_path}")

if __name__ == "__main__":
    main()

