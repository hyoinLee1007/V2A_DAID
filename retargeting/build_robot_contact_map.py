"""Carry the hand half of the CHOIR anchors onto the robot hand.

The object half of a contact map says *where on the cup* was touched. It cannot
say what touched it. Threading a finger through a mug handle puts the pad and
the back of that finger against the same ring of object vertices, so an
object-only map scores the demonstrated grasp and the back-of-hand grasp
identically — which is why reward shaping on it never fixed the latter.

This builds the missing half. Each MANO vertex that took part in an anchor is
pushed through the MANO->robot correspondence, giving a list of

    (robot body id, offset in that body's frame, weight, finger, segment)

Evaluating ``xpos[body] + xmat[body] @ offset`` on the live simulation puts
those points wherever the robot's hand currently is, so a reward can ask "is
the robot touching the cup with *these* patches of skin" rather than only "is
some fingertip near the cup".

    python build_robot_contact_map.py \
        --contact ../reconstruction/heatmap_out/contact_heatmap_palmar.npz \
        --map outputs/mano_robot_map_left.npz

The input contact map should have been extracted with ``--palmar-only``.
Without it roughly 45% of the anchors sit on the back of the hand, and those
would be written in here as *targets* — the opposite of the intent.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", required=True,
                   help="npz from extract_contact_map.py (needs hand_confidence).")
    p.add_argument("--map", default="outputs/mano_robot_map_left.npz")
    p.add_argument("--min-weight", type=float, default=0.05,
                   help="Drop vertices below this share of the peak count. Keeps "
                        "single stray frames out of the target set.")
    p.add_argument("--out", default="outputs/robot_contact_map_{side}.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    c = np.load(args.contact, allow_pickle=True)
    m = np.load(args.map, allow_pickle=True)
    side = str(m["side"])

    if "hand_confidence" not in c.files:
        raise SystemExit(
            f"{args.contact} has no hand_confidence; re-run extract_contact_map.py")
    if not bool(c["palmar_only"]) if "palmar_only" in c.files else False:
        print("WARNING: this contact map was extracted without --palmar-only, so "
              "it contains back-of-hand anchors. They will become targets.")

    Hh = np.asarray(c["hand_confidence"], float)
    body_id = m["body_id"].astype(np.int64)
    offset = np.asarray(m["local_offset"], float)
    finger, segment = m["finger"].astype(str), m["segment"].astype(str)
    palmar = np.asarray(m["mano_palmar"], float)
    side_ok = m["side_ok"].astype(bool)

    keep = (Hh > args.min_weight) & (body_id >= 0)
    # A vertex whose side flipped in the transfer would place a palm-side target
    # on the robot's knuckles. There are few of them and they are exactly the
    # ones this map must not get wrong.
    flipped = int((keep & ~side_ok).sum())
    keep &= side_ok

    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        raise SystemExit("no vertices survived; lower --min-weight")

    w = Hh[idx]
    print(f"MANO vertices in contact : {int((Hh > 0).sum())}")
    print(f"  above min-weight {args.min_weight:<5}: {len(idx)}"
          + (f"   ({flipped} dropped for a flipped side)" if flipped else ""))
    print(f"  weight: mean {w.mean():.3f}  max {w.max():.3f}")

    print(f"\n{'finger':8s} {'pts':>4s} {'weight':>8s} {'share':>7s}   segments")
    tot = w.sum()
    for fg in ("thumb", "index", "middle", "ring", "pinky", "palm"):
        sel = idx[finger[idx] == fg]
        if len(sel) == 0:
            continue
        sw = Hh[sel].sum()
        segs = Counter(segment[sel].tolist())
        print(f"{fg:8s} {len(sel):4d} {sw:8.2f} {100 * sw / tot:6.1f}%   "
              + " ".join(f"{k}:{v}" for k, v in sorted(segs.items())))

    print(f"\npalmar coordinate of the kept vertices: "
          f"min {palmar[idx].min():+.2f}  median {np.median(palmar[idx]):+.2f}")

    # Per-point target lists, carried over from the anchor pairs. Without them
    # the reward sends every skin point to the nearest touched object vertex,
    # so two fingers that must hold different parts of the handle are free to
    # pile onto the same one — measured: index and middle sat at 8.81 and
    # 7.19 cm from "their" target, i.e. the same target. Ragged, stored flat:
    # point k owns tgt_vertex[tgt_offset[k]:tgt_offset[k + 1]].
    tgt_offset = np.zeros(len(idx) + 1, np.int64)
    tgt_vertex = np.zeros(0, np.int64)
    tgt_weight = np.zeros(0, np.float32)
    if "pair_hand" in c.files and len(c["pair_hand"]):
        ph = np.asarray(c["pair_hand"], np.int64)
        po = np.asarray(c["pair_object"], np.int64)
        pw = np.asarray(c["pair_weight"], np.float32)
        order = np.argsort(ph, kind="stable")
        ph, po, pw = ph[order], po[order], pw[order]
        starts = np.searchsorted(ph, idx, "left")
        ends = np.searchsorted(ph, idx, "right")
        pieces_v, pieces_w = [], []
        for n, (s, e) in enumerate(zip(starts, ends)):
            pieces_v.append(po[s:e])
            pieces_w.append(pw[s:e])
            tgt_offset[n + 1] = tgt_offset[n] + (e - s)
        tgt_vertex = np.concatenate(pieces_v) if pieces_v else tgt_vertex
        tgt_weight = np.concatenate(pieces_w) if pieces_w else tgt_weight
        counts = np.diff(tgt_offset)
        orphan = int((counts == 0).sum())
        print(f"\ntarget lists: {len(tgt_vertex)} pairs over {len(idx)} points  "
              f"(median {int(np.median(counts))} targets/point, "
              f"min {counts.min()}, max {counts.max()}"
              + (f", {orphan} points with none" if orphan else "") + ")")
    else:
        print("\nWARNING: this contact map has no pair_hand/pair_object; "
              "re-run extract_contact_map.py to get per-point targets.")

    out = Path(args.out.format(side=side))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             tgt_offset=tgt_offset, tgt_vertex=tgt_vertex, tgt_weight=tgt_weight,
             body_id=body_id[idx], local_offset=offset[idx], weight=w,
             finger=finger[idx], segment=segment[idx], mano_vertex=idx,
             body_name=m["body_name"].astype(str)[idx],
             side=side, scene=str(m["scene"]), contact_src=str(args.contact))
    print(f"\nSaved {out}   ({len(idx)} robot-side contact points)")


if __name__ == "__main__":
    main()
