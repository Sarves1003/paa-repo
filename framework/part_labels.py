#!/usr/bin/env python3
"""Check or correct the part condition recorded with every placement of the view library.

  python3 part_labels.py list
  python3 part_labels.py set --placement horizontal_004 --condition no_ring_open --rust-facing none
  python3 part_labels.py show --placement horizontal_004        # writes results_task/check_<placement>.jpg
"""
import argparse, glob, json, os
import cv2
import numpy as np
import task_model as TM

VIEWS = os.path.join("deploy", "views")


def load(d):
    return json.load(open(os.path.join(d, "meta.json")))


def cmd_list(a):
    rows = []
    for d in sorted(glob.glob(os.path.join(VIEWS, "*_*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        m = load(d)
        cond = m.get("condition")
        ok = sum(1 for v in m["views"] if v.get("ok"))
        rows.append((os.path.basename(d), m["pose"], cond or "-- MISSING --", m.get("rust_facing", "none"), ok,
                     TM.describe(TM.labels(cond, m.get("rust_facing", "none"))[0]) if cond else ""))
    print("%-16s %-10s %-24s %-6s %5s  %s" % ("placement", "pose", "condition", "rust", "views", "verdict"))
    for r in rows:
        print("%-16s %-10s %-24s %-6s %5d  %s" % r)
    from collections import Counter
    c = Counter((r[1], r[2]) for r in rows)
    print("\nplacements per pose and condition:")
    for (pz, cd), n in sorted(c.items()):
        print("  %-10s %-24s %d" % (pz, cd, n))
    miss = [r[0] for r in rows if r[2].startswith("--")]
    if miss:
        print("\n%d placement(s) have no condition: python3 part_labels.py set --placement <name> --condition ..." % len(miss))


def cmd_set(a):
    d = os.path.join(VIEWS, a.placement)
    m = load(d)
    try:
        y, mk = TM.labels(a.condition, a.rust_facing)
    except ValueError as e:
        raise SystemExit(str(e))
    old = m.get("condition")
    m.update(condition=a.condition, rust_facing=a.rust_facing, task_items=TM.ITEMS,
             task_labels=[int(v) for v in y], task_mask=[int(v) for v in mk])
    json.dump(m, open(os.path.join(d, "meta.json"), "w"), indent=2, default=float)
    print("%s: %s -> %s (%s)" % (a.placement, old, a.condition, TM.describe(y)))


def cmd_show(a):
    d = os.path.join(VIEWS, a.placement)
    m = load(d)
    sv = cv2.imread(os.path.join(d, "survey.jpg"))
    imgs = [sv] + [cv2.imread(os.path.join(d, "v%02d.jpg" % v["k"])) for v in m["views"] if v.get("ok")][:7]
    imgs = [cv2.resize(i, (320, 240)) for i in imgs if i is not None]
    while len(imgs) % 4:
        imgs.append(np.zeros_like(imgs[0]))
    sheet = np.vstack([np.hstack(imgs[i:i + 4]) for i in range(0, len(imgs), 4)])
    cv2.putText(sheet, "%s: %s" % (a.placement, m.get("condition")), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    os.makedirs("results_task", exist_ok=True)
    f = os.path.join("results_task", "check_%s.jpg" % a.placement); cv2.imwrite(f, sheet); print("wrote", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "set", "show"])
    ap.add_argument("--placement")
    ap.add_argument("--condition")
    ap.add_argument("--rust-facing", dest="rust_facing", choices=["none", "up", "side", "down"], default="none")
    a = ap.parse_args()
    {"list": cmd_list, "set": cmd_set, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
