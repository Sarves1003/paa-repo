#!/usr/bin/env python3
"""Convert between NumPy .npz bundles and plain (non-archived) folders.

The repository stores every array as an individual .npy file (and the part crops as PNG images), because
archived files are not allowed. The analysis and training code reads .npz bundles, so run once after cloning:

    python3 tools/npz_tool.py rebuild          # creates every data/*.npz and models/*.npz from its folder

Folder layout for a bundle NAME.npz:  NAME/ with one KEY.npy per array, keys.txt (array order) and, for the
part crops, NAME/X_png/00000.png ... (the array X stored image by image, channel order unchanged).
"""
import glob, os, sys
import numpy as np


def unpack(npz_path):
    out = npz_path[:-4]
    os.makedirs(out, exist_ok=True)
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.files)
    for k in keys:
        a = z[k]
        if k == "X" and a.dtype == np.uint8 and a.ndim == 4:      # images: store as lossless PNG files
            from PIL import Image
            d = os.path.join(out, "X_png"); os.makedirs(d, exist_ok=True)
            for i, im in enumerate(a):
                Image.fromarray(im).save(os.path.join(d, "%05d.png" % i))
        else:
            np.save(os.path.join(out, k + ".npy"), a, allow_pickle=False)
    open(os.path.join(out, "keys.txt"), "w").write("\n".join(keys) + "\n")
    return out


def rebuild_one(folder):
    keys = [k for k in open(os.path.join(folder, "keys.txt")).read().split() if k]
    arrays = {}
    for k in keys:
        if k == "X" and os.path.isdir(os.path.join(folder, "X_png")):
            from PIL import Image
            files = sorted(glob.glob(os.path.join(folder, "X_png", "*.png")))
            arrays[k] = np.stack([np.asarray(Image.open(f)) for f in files]).astype(np.uint8)
        else:
            arrays[k] = np.load(os.path.join(folder, k + ".npy"), allow_pickle=False)
    np.savez_compressed(folder + ".npz", **arrays)
    return folder + ".npz"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rebuild"
    if cmd == "unpack":
        for p in sys.argv[2:]:
            print("unpacked", unpack(p))
    elif cmd == "rebuild":
        for f in sorted(glob.glob("**/keys.txt", recursive=True)):
            print("rebuilt", rebuild_one(os.path.dirname(f)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
