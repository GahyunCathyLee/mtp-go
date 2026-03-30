#!/usr/bin/env python3
"""
preprocess.py — highD → MTP-GO format with importance features

Node feature schema (N_IN = 7):
  Node 0 (ego):     [x−ref_x, y−ref_y, vx,  vy,  ax,  ay,  −1.0 ]
  Node k (nb slot): [dx,      dy,      dvx, dvy, dax, day, I_feat]
  ref_x, ref_y = ego position at last observation frame (t0_frame)
  Absent neighbor slots are NaN-filled.

Target schema (N_OUT = 6):
  Node 0 (ego):     [x_fut−ref_x, y_fut−ref_y, vx, vy, ax, ay]
  Node k (nb slot): [dx_fut,      dy_fut,      dvx, dvy, dax, day]

Global shifts applied (same as neighformer preprocess):
  1) Recording-level: x -= x_min, y -= y_min
  2) Sample-level:    ref_x = ego_x at last history frame

Output: data/highD-imp-gnn/{training,validation,testing}/
Split:  rec 01-43 train | 44-51 val | 52-60 test  (configurable)
"""
from __future__ import annotations

import argparse
import bisect
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NEIGHBOR_COLS_8 = [
    "precedingId", "followingId",
    "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
    "rightPrecedingId", "rightAlongsideId", "rightFollowingId",
]
K     = 8   # fixed neighbor slots
N_IN  = 7   # node feature dim (history)
N_OUT = 6   # node feature dim (target, no importance)

LIS_BINS: Dict[str, Dict] = {
    '3': {'cuts': [-5.8639, 4.9525],
          'vals': [-1.0, 0.0, 1.0]},
    '5': {'cuts': [-13.7033, -3.0238, 2.2735, 13.0957],
          'vals': [-2., -1., 0., 1., 2.]},
    '7': {'cuts': [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
          'vals': [-3., -2., -1., 0., 1., 2., 3.]},
    '9': {'cuts': [-22.7661, -12.1209, -5.8639, -1.4829, 0.9127, 4.9525, 11.4115, 22.7702],
          'vals': [-4., -3., -2., -1., 0., 1., 2., 3., 4.]},
}
IMP_LIS = {'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
            'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5}
IMP_LIT = {'sx': 15.0, 'ax': 0.2, 'bx': 0.25,
            'sy':  2.0, 'ay': 0.01, 'by': 0.1}

SPLIT_DEFAULT = {
    'training':   (1,  43),
    'validation': (44, 51),
    'testing':    (52, 60),
}


# ─────────────────────────────────────────────────────────────────────────────
# MetaInfo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaInfo:
    rec_id: str
    frame: int
    initial_pos: list
    vehicle_ids: list
    vehicle_types: list
    euclidian_dist: list
    maneuver_id: list
    width: list
    length: list


# ─────────────────────────────────────────────────────────────────────────────
# LIS / Importance
# ─────────────────────────────────────────────────────────────────────────────

def _lit_to_lis(lit: float, lis_mode: str) -> float:
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]


def _compute_importance_lis(lis, delta_lane, lc_state):
    p = IMP_LIS
    ix = (np.exp(-(lis**2) / (2*p['sx']**2))
          * np.exp(-p['ax'] * lc_state)
          * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state**2) / (2*p['sy']**2))
          * np.exp(-p['ay'] * abs(lis)**p['py'])
          * np.exp(-p['by'] * delta_lane))
    i  = float(np.sqrt((ix**2 + iy**2) / 2.0))
    return float(ix), float(iy), i


def _compute_importance_lit(lit, delta_lane, lc_state):
    p = IMP_LIT
    ix = (np.exp(-(lit**2) / (2*p['sx']**2))
          * np.exp(-p['ax'] * lc_state)
          * np.exp(-p['bx'] * delta_lane))
    iy = (np.exp(-(lc_state**2) / (2*p['sy']**2))
          * np.exp(-p['ay'] * abs(lit)**1.5)
          * np.exp(-p['by'] * delta_lane))
    i  = float(np.sqrt((ix**2 + iy**2) / 2.0))
    return float(ix), float(iy), i


# ─────────────────────────────────────────────────────────────────────────────
# Recording-level helpers  (mirrored from neighformer preprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_semicolon_floats(s) -> List[float]:
    if not isinstance(s, str):
        return []
    return [float(p) for p in s.strip().split(';') if p.strip()]


def _flip_constants(rec_meta: pd.DataFrame):
    fr    = float(rec_meta.loc[0, 'frameRate'])
    upper = _parse_semicolon_floats(str(rec_meta.loc[0, 'upperLaneMarkings']))
    lower = _parse_semicolon_floats(str(rec_meta.loc[0, 'lowerLaneMarkings']))
    ua, la = np.array(upper, np.float32), np.array(lower, np.float32)
    C_y   = float(ua[-1] + la[0]) if (len(ua) and len(la)) else 0.0
    return C_y, fr, ua, la


def _maybe_flip(x, y, xv, yv, xa, ya, lane_id, dd, C_y, x_max, upper_mm):
    mask = (dd == 1)
    if not np.any(mask):
        return x, y, xv, yv, xa, ya, lane_id
    x2, y2, xv2, yv2, xa2, ya2, l2 = (a.copy() for a in (x, y, xv, yv, xa, ya, lane_id))
    x2[mask]  = x_max - x2[mask]
    y2[mask]  = C_y   - y2[mask]
    xv2[mask] = -xv2[mask];  yv2[mask] = -yv2[mask]
    xa2[mask] = -xa2[mask];  ya2[mask] = -ya2[mask]
    if upper_mm is not None:
        mn, mx_v = upper_mm
        ok = mask & (l2 > 0)
        l2[ok] = (mn + mx_v) - l2[ok]
    return x2, y2, xv2, yv2, xa2, ya2, l2


def _build_lane_tables(markings: np.ndarray):
    if markings is None or len(markings) < 2:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    left, right = markings[:-1], markings[1:]
    return ((right + left) * 0.5).astype(np.float32), (right - left).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# lc_state  (v3: latV + lane-centre-offset based)
# ─────────────────────────────────────────────────────────────────────────────

def _lc_state_v3(ki: int, nb_lat_v: float, nb_lco: float) -> float:
    if ki < 2:   # same-lane slot (preceding / following)
        if (nb_lco < -1.0 and nb_lat_v > 0.0) or (nb_lco > 1.0 and nb_lat_v < 0.0):
            return 0.0
        elif (nb_lco < -1.0 and nb_lat_v < 0.0) or (nb_lco > 1.0 and nb_lat_v > 0.0) \
                or abs(nb_lat_v) > 0.029:
            return 2.0
        else:
            return 1.0
    elif ki < 5:  # left-lane slots  (2, 3, 4)
        if   nb_lat_v < -0.029: return 0.0
        elif nb_lat_v >  0.029: return 2.0
        else:                   return 1.0
    else:         # right-lane slots (5, 6, 7)
        if   nb_lat_v < -0.029: return 2.0
        elif nb_lat_v >  0.029: return 0.0
        else:                   return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Edge index / feature builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_edges(nb_dx: np.ndarray, nb_dy: np.ndarray,
                 nb_present: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build fully-connected edge_index and Euclidean edge_feat for one timestep.

    Node 0 = ego (always present, ego-relative position = (0, 0)).
    Nodes 1..K = neighbor slots; absent slots are skipped.

    nb_dx, nb_dy : (K,) ego-relative dx/dy for each slot
    nb_present   : (K,) bool — True if slot is active
    """
    # Collect valid node indices and their ego-relative positions
    valid_idx  = [0]           # ego always present
    positions  = [[0.0, 0.0]]  # ego at origin

    for k in range(K):
        if nb_present[k]:
            valid_idx.append(k + 1)
            positions.append([float(nb_dx[k]), float(nb_dy[k])])

    n = len(valid_idx)
    if n < 2:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 1), dtype=torch.float32))

    pos = np.array(positions, np.float32)   # (n, 2)
    srcs, dsts, dists = [], [], []
    for i in range(n):
        for j in range(n):
            d = float(np.sqrt(np.sum((pos[i] - pos[j])**2)))
            srcs.append(valid_idx[i])
            dsts.append(valid_idx[j])
            dists.append(d)

    ei = torch.tensor([srcs, dsts], dtype=torch.long)
    ef = torch.tensor(dists, dtype=torch.float32).unsqueeze(1)
    return ei, ef


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording processing
# ─────────────────────────────────────────────────────────────────────────────

def process_recording(rec_id: str, args) -> Optional[List[dict]]:
    raw_dir = Path(args.data_dir)
    try:
        rec_meta = pd.read_csv(raw_dir / f"{rec_id}_recordingMeta.csv")
        trk_meta = pd.read_csv(raw_dir / f"{rec_id}_tracksMeta.csv")
        tracks   = pd.read_csv(raw_dir / f"{rec_id}_tracks.csv")
    except FileNotFoundError:
        print(f"  [WARN] rec {rec_id} not found, skipping.")
        return None

    # Column aliases
    for df in (trk_meta, tracks):
        if 'id' in df.columns and 'trackId' not in df.columns:
            df.rename(columns={'id': 'trackId'}, inplace=True)

    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns:
            tracks[c] = 0
    for c in ('xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration'):
        if c not in tracks.columns:
            tracks[c] = 0.0
    if 'laneId' not in tracks.columns:
        tracks['laneId'] = 0

    # ── Recording constants ───────────────────────────────────────────────────
    C_y, frame_rate, upper_mark, lower_mark = _flip_constants(rec_meta)
    step   = max(1, int(round(frame_rate / args.target_hz)))
    T      = int(round(args.history_sec  * args.target_hz))
    Tf     = int(round(args.future_sec   * args.target_hz))
    stride = max(1, int(round(args.stride_sec * args.target_hz)))

    # ── Vehicle meta lookup ───────────────────────────────────────────────────
    vid_to_dd = dict(zip(trk_meta['trackId'].astype(int),
                         trk_meta['drivingDirection'].astype(int)))
    # highD: 'width' = longitudinal length, 'height' = lateral width
    vid_to_lon = dict(zip(trk_meta['trackId'].astype(int),
                          trk_meta['width'].astype(float)))   # longitudinal (for LIT gap)
    vid_to_lat = dict(zip(trk_meta['trackId'].astype(int),
                          trk_meta['height'].astype(float)))  # lateral (for MetaInfo.width)
    vid_to_class: Dict[int, str] = {}
    for _, row in trk_meta.iterrows():
        v   = int(row['trackId'])
        cls = str(row.get('class', 'Car'))
        vid_to_class[v] = cls if cls in ('Car', 'Truck') else 'Car'

    # ── Raw arrays ────────────────────────────────────────────────────────────
    frame   = tracks['frame'].astype(np.int32).to_numpy()
    vid     = tracks['trackId'].astype(np.int32).to_numpy()
    xraw    = tracks['x'].astype(np.float32).to_numpy().copy()
    yraw    = tracks['y'].astype(np.float32).to_numpy().copy()

    # Shift to vehicle center (top-left corner → center)
    lon_row = np.array([vid_to_lon.get(int(v_), 0.) for v_ in vid], np.float32)
    lat_row = np.array([vid_to_lat.get(int(v_), 0.) for v_ in vid], np.float32)
    xraw += 0.5 * lon_row
    yraw += 0.5 * lat_row

    xv      = tracks['xVelocity'].astype(np.float32).to_numpy()
    yv      = tracks['yVelocity'].astype(np.float32).to_numpy()
    xa      = tracks['xAcceleration'].astype(np.float32).to_numpy()
    ya      = tracks['yAcceleration'].astype(np.float32).to_numpy()
    lane_id = tracks['laneId'].astype(np.int16).to_numpy()
    dd      = np.array([vid_to_dd.get(int(v_), 0) for v_ in vid], np.int8)
    x_max   = float(np.nanmax(xraw)) if xraw.size else 0.0

    # ── Lane-centre offset (for lc_state v3) ─────────────────────────────────
    _N_upper           = len(upper_mark)
    lat_lane_offset    = np.zeros(len(yraw), np.float32)
    _lid_arr           = lane_id.astype(np.int32)

    _mask_lo = (dd == 2)
    _j_lo    = _lid_arr - _N_upper - 2
    _ok_lo   = _mask_lo & (_j_lo >= 0) & (_j_lo < len(lower_mark) - 1)
    lat_lane_offset[_ok_lo] = (
        yraw[_ok_lo]
        - 0.5 * (lower_mark[_j_lo[_ok_lo]] + lower_mark[_j_lo[_ok_lo] + 1])
    )
    _mask_up = (dd == 1)
    _j_up    = _lid_arr - 2
    _ok_up   = _mask_up & (_j_up >= 0) & (_j_up < len(upper_mark) - 1)
    lat_lane_offset[_ok_up] = (
        yraw[_ok_up]
        - 0.5 * (upper_mark[_j_up[_ok_up]] + upper_mark[_j_up[_ok_up] + 1])
    )
    lat_lane_offset[dd == 1] *= -1.0   # negate for upper-dir after flip

    # ── Flip upper-direction vehicles & recording-level shift ─────────────────
    upper_for_calc = upper_mark.copy()
    if args.normalize_upper_xy and len(upper_for_calc):
        upper_for_calc = np.sort((C_y - upper_for_calc).astype(np.float32))
    upper_mm = (1, int(len(_build_lane_tables(upper_for_calc)[0]))) \
               if len(upper_for_calc) >= 2 else None

    if args.normalize_upper_xy:
        xraw, yraw, xv, yv, xa, ya, lane_id = _maybe_flip(
            xraw, yraw, xv, yv, xa, ya, lane_id, dd, C_y, x_max, upper_mm
        )

    x_min = float(np.nanmin(xraw)) if xraw.size else 0.0
    y_min = float(np.nanmin(yraw)) if yraw.size else 0.0
    x = (xraw - x_min).astype(np.float32)
    y = (yraw - y_min).astype(np.float32)

    # ── Per-vehicle row-index lookup ──────────────────────────────────────────
    per_vid_rows: Dict[int, np.ndarray]       = {}
    per_vid_f2r:  Dict[int, Dict[int, int]]   = {}
    for v_, idxs in tracks.groupby('trackId').indices.items():
        idxs = np.array(idxs, np.int32)
        idxs = idxs[np.argsort(frame[idxs])]
        per_vid_rows[int(v_)] = idxs
        per_vid_f2r[int(v_)]  = {int(frame[r]): int(r) for r in idxs}

    # Neighbour-ID table  shape (n_rows, K)
    nb_ids_all = np.stack(
        [tracks[c].astype(np.int32).to_numpy() for c in NEIGHBOR_COLS_8], axis=1
    )

    imp_feat_idx = {'I_x': 0, 'I_y': 1, 'I': 2}[args.importance_feat]
    samples: List[dict] = []

    # ── Iterate over ego vehicles ─────────────────────────────────────────────
    for v_, idxs in per_vid_rows.items():
        frs = frame[idxs]
        if len(frs) < (T + Tf) * step:
            continue
        fr_set    = set(map(int, frs.tolist()))
        start_min = int(frs[0]  + (T - 1) * step)
        end_max   = int(frs[-1] - Tf       * step)
        if start_min > end_max:
            continue

        len_ego = float(vid_to_lon.get(v_, 0.0))

        t0_frame = start_min
        while t0_frame <= end_max:
            hist_frames = [t0_frame - (T - 1 - i) * step for i in range(T)]
            fut_frames  = [t0_frame + (i + 1) * step      for i in range(Tf)]

            if (not all(hf in fr_set for hf in hist_frames) or
                    not all(ff in fr_set for ff in fut_frames)):
                t0_frame += stride * step
                continue

            # ── Ego history ───────────────────────────────────────────────────
            h_rows = [per_vid_f2r[v_][hf] for hf in hist_frames]
            f_rows = [per_vid_f2r[v_][ff] for ff in fut_frames]

            ex   = x[h_rows];  ey   = y[h_rows]
            exv  = xv[h_rows]; eyv  = yv[h_rows]
            exa  = xa[h_rows]; eya  = ya[h_rows]
            e_lid = lane_id[h_rows].astype(np.int32)

            # Sample-level shift: last history frame as origin
            ref_x = float(ex[-1])
            ref_y = float(ey[-1])

            ex_f  = x[f_rows];  ey_f  = y[f_rows]
            exv_f = xv[f_rows]; eyv_f = yv[f_rows]
            exa_f = xa[f_rows]; eya_f = ya[f_rows]

            # ── Node tensors ──────────────────────────────────────────────────
            inp = np.full((K + 1, T,  N_IN),  np.nan, np.float32)
            tgt = np.full((K + 1, Tf, N_OUT), np.nan, np.float32)

            # Ego node (index 0)
            inp[0, :, 0] = ex  - ref_x
            inp[0, :, 1] = ey  - ref_y
            inp[0, :, 2] = exv
            inp[0, :, 3] = eyv
            inp[0, :, 4] = exa
            inp[0, :, 5] = eya
            inp[0, :, 6] = -1.0   # dummy importance

            tgt[0, :, 0] = ex_f  - ref_x
            tgt[0, :, 1] = ey_f  - ref_y
            tgt[0, :, 2] = exv_f
            tgt[0, :, 3] = eyv_f
            tgt[0, :, 4] = exa_f
            tgt[0, :, 5] = eya_f

            # ── Determine neighbor vehicle IDs from t0_frame ──────────────────
            r0     = per_vid_f2r[v_][t0_frame]
            nb_ids = [int(nb_ids_all[r0, ki]) for ki in range(K)]

            # For MetaInfo
            nb_vids    = nb_ids[:]
            nb_classes = [vid_to_class.get(nid, 'Car') if nid > 0 else 'Car'
                          for nid in nb_ids]
            nb_widths  = [vid_to_lat.get(nid, 0.) if nid > 0 else 0.
                          for nid in nb_ids]      # lateral width
            nb_lengths = [vid_to_lon.get(nid, 0.) if nid > 0 else 0.
                          for nid in nb_ids]      # longitudinal length

            # Storage for edge building
            nb_dx_hist = np.full((T,  K), np.nan, np.float32)
            nb_dy_hist = np.full((T,  K), np.nan, np.float32)
            nb_ok_hist = np.zeros((T,  K), bool)
            nb_dx_fut  = np.full((Tf, K), np.nan, np.float32)
            nb_dy_fut  = np.full((Tf, K), np.nan, np.float32)
            nb_ok_fut  = np.zeros((Tf, K), bool)

            # ── Neighbor nodes (indices 1..K) ─────────────────────────────────
            for ki, nid in enumerate(nb_ids):
                if nid <= 0:
                    continue
                rm = per_vid_f2r.get(nid)
                if rm is None:
                    continue

                len_nb = float(vid_to_lon.get(nid, 0.0))

                # History
                for ti, hf in enumerate(hist_frames):
                    r = rm.get(int(hf))
                    if r is None:
                        continue

                    nb_x, nb_y   = float(x[r]),  float(y[r])
                    nb_xv, nb_yv = float(xv[r]), float(yv[r])
                    nb_xa, nb_ya = float(xa[r]), float(ya[r])
                    eg_x, eg_y   = float(ex[ti]), float(ey[ti])

                    dx_  = nb_x  - eg_x
                    dy_  = nb_y  - eg_y
                    dvx_ = nb_xv - float(exv[ti])
                    dvy_ = nb_yv - float(eyv[ti])
                    dax_ = nb_xa - float(exa[ti])
                    day_ = nb_ya - float(eya[ti])

                    inp[ki + 1, ti, 0] = dx_
                    inp[ki + 1, ti, 1] = dy_
                    inp[ki + 1, ti, 2] = dvx_
                    inp[ki + 1, ti, 3] = dvy_
                    inp[ki + 1, ti, 4] = dax_
                    inp[ki + 1, ti, 5] = day_

                    nb_dx_hist[ti, ki] = dx_
                    nb_dy_hist[ti, ki] = dy_
                    nb_ok_hist[ti, ki] = True

                    # lc_state v3
                    lc_state = _lc_state_v3(ki,
                                            float(yv[r]),
                                            float(lat_lane_offset[r]))

                    # LIT → LIS
                    half_sum   = 0.5 * (len_ego + len_nb)
                    denom_base = dvx_ if dx_ >= 0 else -dvx_
                    gap        = abs(dx_) - half_sum
                    gap        = max(gap, 0.0)
                    eps        = args.eps_gate
                    lit        = gap / (denom_base + (eps if denom_base >= 0 else -eps))
                    lis        = _lit_to_lis(lit, args.lis_mode)
                    delta_lane = float(abs(int(lane_id[r]) - int(e_lid[ti])))

                    if args.importance_mode == 'lit':
                        ix, iy, i_total = _compute_importance_lit(lit, delta_lane, lc_state)
                    else:
                        ix, iy, i_total = _compute_importance_lis(lis, delta_lane, lc_state)

                    imp_vals = (ix, iy, i_total)
                    inp[ki + 1, ti, 6] = float(imp_vals[imp_feat_idx])

                # Future
                for fi, ff in enumerate(fut_frames):
                    r = rm.get(int(ff))
                    if r is None:
                        continue

                    nb_x_f, nb_y_f   = float(x[r]),  float(y[r])
                    nb_xv_f, nb_yv_f = float(xv[r]), float(yv[r])
                    nb_xa_f, nb_ya_f = float(xa[r]), float(ya[r])

                    dx_f  = nb_x_f  - float(ex_f[fi])
                    dy_f  = nb_y_f  - float(ey_f[fi])
                    dvx_f = nb_xv_f - float(exv_f[fi])
                    dvy_f = nb_yv_f - float(eyv_f[fi])
                    dax_f = nb_xa_f - float(exa_f[fi])
                    day_f = nb_ya_f - float(eya_f[fi])

                    tgt[ki + 1, fi, 0] = dx_f
                    tgt[ki + 1, fi, 1] = dy_f
                    tgt[ki + 1, fi, 2] = dvx_f
                    tgt[ki + 1, fi, 3] = dvy_f
                    tgt[ki + 1, fi, 4] = dax_f
                    tgt[ki + 1, fi, 5] = day_f

                    nb_dx_fut[fi, ki] = dx_f
                    nb_dy_fut[fi, ki] = dy_f
                    nb_ok_fut[fi, ki] = True

            # ── Build edge indices / features ──────────────────────────────────
            hist_ei, hist_ef = [], []
            for ti in range(T):
                ei, ef = _build_edges(nb_dx_hist[ti], nb_dy_hist[ti], nb_ok_hist[ti])
                hist_ei.append(ei)
                hist_ef.append(ef)

            fut_ei, fut_ef = [], []
            for fi in range(Tf):
                ei, ef = _build_edges(nb_dx_fut[fi], nb_dy_fut[fi], nb_ok_fut[fi])
                fut_ei.append(ei)
                fut_ef.append(ef)

            # ── MetaInfo ──────────────────────────────────────────────────────
            nb_dists = [0.0]   # ego-to-ego = 0
            for ki in range(K):
                if nb_ok_hist[-1, ki]:
                    nb_dists.append(float(np.sqrt(
                        nb_dx_hist[-1, ki]**2 + nb_dy_hist[-1, ki]**2)))
                else:
                    nb_dists.append(float('nan'))

            meta = MetaInfo(
                rec_id       = rec_id,
                frame        = t0_frame,
                initial_pos  = [ref_x, ref_y],
                vehicle_ids  = [v_] + nb_vids,
                vehicle_types= [vid_to_class.get(v_, 'Car')] + nb_classes,
                euclidian_dist = nb_dists,
                maneuver_id  = [3] * (K + 1),   # dummy; not used by model
                width        = [vid_to_lat.get(v_, 0.)] + nb_widths,
                length       = [vid_to_lon.get(v_, 0.)] + nb_lengths,
            )

            samples.append({
                'inp':     torch.from_numpy(inp).float(),
                'tgt':     torch.from_numpy(tgt).float(),
                'hist_ei': hist_ei,
                'hist_ef': hist_ef,
                'fut_ei':  fut_ei,
                'fut_ef':  fut_ef,
                'meta':    meta,
            })

            t0_frame += stride * step

    return samples if samples else None


# ─────────────────────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────────────────────

def create_directories(out_root: Path, overwrite: bool) -> None:
    splits = ('training', 'validation', 'testing')
    subdirs = ('observation', 'target', 'meta')
    for split in splits:
        for sub in subdirs:
            d = out_root / split / sub
            if d.exists() and overwrite:
                shutil.rmtree(out_root / split)
                break
        for sub in subdirs:
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Save one sample
# ─────────────────────────────────────────────────────────────────────────────

def save_sample(sample: dict, split: str, idx: int, out_root: Path) -> None:
    inp  = sample['inp']   # (K+1, T,  N_IN)
    tgt  = sample['tgt']   # (K+1, Tf, N_OUT)

    nan_mask  = torch.isnan(inp)
    real_mask = ~torch.isnan(tgt)

    inp_clean = inp.clone();  inp_clean[nan_mask]          = 0.0
    tgt_clean = tgt.clone();  tgt_clean[~real_mask]        = 0.0

    obs_dir  = out_root / split / 'observation'
    tar_dir  = out_root / split / 'target'
    meta_dir = out_root / split / 'meta'

    torch.save(inp_clean,          obs_dir / f'dat{idx}.pt')
    torch.save(nan_mask,           obs_dir / f'nan_mask{idx}.pt')
    torch.save(sample['hist_ei'],  obs_dir / f'edge_idx{idx}.pt')
    torch.save(sample['hist_ef'],  obs_dir / f'edge_feat{idx}.pt')

    torch.save(tgt_clean,          tar_dir / f'dat{idx}.pt')
    torch.save(real_mask,          tar_dir / f'real_mask{idx}.pt')
    torch.save(sample['fut_ei'],   tar_dir / f'edge_idx{idx}.pt')
    torch.save(sample['fut_ef'],   tar_dir / f'edge_feat{idx}.pt')

    torch.save(sample['meta'],     meta_dir / f'dat{idx}.pt')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _rec_id_str(n: int) -> str:
    return f'{n:02d}'


def main(args) -> None:
    out_root = Path(args.out_dir)
    create_directories(out_root, args.overwrite)

    # Determine split ranges
    train_range = range(args.train_start, args.train_end + 1)
    val_range   = range(args.val_start,   args.val_end   + 1)
    test_range  = range(args.test_start,  args.test_end  + 1)

    split_map = {}
    for n in train_range: split_map[_rec_id_str(n)] = 'training'
    for n in val_range:   split_map[_rec_id_str(n)] = 'validation'
    for n in test_range:  split_map[_rec_id_str(n)] = 'testing'

    counters = {'training': 0, 'validation': 0, 'testing': 0}
    id_lists = {'training': [], 'validation': [], 'testing': []}

    all_rec_ids = sorted(split_map.keys())
    for rec_id in tqdm(all_rec_ids, desc='Recordings'):
        split   = split_map[rec_id]
        samples = process_recording(rec_id, args)
        if not samples:
            continue
        for s in samples:
            idx = counters[split]
            save_sample(s, split, idx, out_root)
            id_lists[split].append(idx)
            counters[split] += 1

    for split in ('training', 'validation', 'testing'):
        torch.save(id_lists[split], out_root / split / 'ids.pt')
        print(f'  {split:12s}: {counters[split]:,} samples')

    print(f'\n[OK] Data written to {out_root}')
    print(f'     Use --dataset highD-imp when running train.py')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='highD → MTP-GO preprocessing with importance features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--data_dir',  default='data/highD/raw',
                    help='Directory containing raw highD CSV files')
    ap.add_argument('--out_dir',   default='data/highD-imp-gnn',
                    help='Output root directory')
    ap.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing output directory')

    # Temporal parameters (should match train.py dt=0.2 → 5 Hz)
    ap.add_argument('--target_hz',   type=float, default=3.0,
                    help='Target sampling rate [Hz] — must match train.py dt (default 5 Hz = dt 0.2s)')
    ap.add_argument('--history_sec', type=float, default=2.0,
                    help='History length [s] — matches MTP-GO input_len=2')
    ap.add_argument('--future_sec',  type=float, default=5.0,
                    help='Prediction horizon [s]')
    ap.add_argument('--stride_sec',  type=float, default=1.0,
                    help='Sliding window stride [s]')

    # Coordinate normalisation
    ap.add_argument('--normalize_upper_xy', action='store_true', default=True,
                    help='Flip upper-direction vehicles to unified coordinate frame')

    # Importance
    ap.add_argument('--importance_feat', default='I_y',
                    choices=['I_x', 'I_y', 'I'],
                    help='Which importance scalar to use as the 7th feature')
    ap.add_argument('--importance_mode', default='lis',
                    choices=['lis', 'lit'],
                    help='Importance computation mode')
    ap.add_argument('--lis_mode',  default='7',
                    choices=['3', '5', '7', '9'],
                    help='LIS binning granularity (used when importance_mode=lis)')
    ap.add_argument('--eps_gate',  type=float, default=1.0,
                    help='Epsilon for LIT denominator stabilisation')

    # Train/val/test split by recording ID
    ap.add_argument('--train_start', type=int, default=SPLIT_DEFAULT['training'][0])
    ap.add_argument('--train_end',   type=int, default=SPLIT_DEFAULT['training'][1])
    ap.add_argument('--val_start',   type=int, default=SPLIT_DEFAULT['validation'][0])
    ap.add_argument('--val_end',     type=int, default=SPLIT_DEFAULT['validation'][1])
    ap.add_argument('--test_start',  type=int, default=SPLIT_DEFAULT['testing'][0])
    ap.add_argument('--test_end',    type=int, default=SPLIT_DEFAULT['testing'][1])

    args = ap.parse_args()
    main(args)
