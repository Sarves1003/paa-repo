#!/usr/bin/env python3
"""Move data between the Jetson Nano (robot) and Colab (heavy computation).

On the Jetson, after collecting data:
    python3 sync.py pack-for-colab          -> to_colab_<stamp>.zip
      contains: config.py, deploy/ (locked pose, grounding, cost models, current policy),
                data/ (survey datasets, reference image), hw_logs/ (logs, jetson_check.json;
                images optional with --with-images)
In Colab, after training/analysis (see colab_pipeline.ipynb):
    python3 sync.py pack-for-jetson         -> to_jetson_<stamp>.zip
      contains: deploy/*.npz (policies, refit cost models), deploy/time_model_*.json,
                and MANIFEST.json (what was trained, on which data, with which results)
Back on the Jetson:
    python3 sync.py unpack to_jetson_<stamp>.zip
      verifies checksums, backs up files it replaces into deploy/backup_<stamp>/,
      and prints the manifest so you know which policy you are deploying.
Standard library only (works on the Jetson's Python 3.6).
"""
import argparse, glob, hashlib, json, os, shutil, sys, time, zipfile


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def add(z, path, manifest):
    if os.path.isfile(path):
        z.write(path, path); manifest["files"][path] = sha(path)


def pack_for_colab(a):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = "to_colab_%s.zip" % stamp
    man = {"direction": "jetson->colab", "created": stamp, "files": {}}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        add(z, "config.py", man)
        for p in glob.glob("deploy/*.json") + glob.glob("deploy/*.npz") + glob.glob("deploy/intrinsics*"):
            add(z, p, man)
        for p in glob.glob("data/*.npz") + glob.glob("data/*.jpg"):
            add(z, p, man)
        for p in glob.glob("deploy/surveys/*.npz") + glob.glob("deploy/surveys/*.jpg"):
            if a.with_images or p.endswith(".npz"):
                add(z, p, man)
        for p in glob.glob("hw_logs/**/*.csv", recursive=True) + glob.glob("hw_logs/**/*.json", recursive=True):
            add(z, p, man)
        if a.with_images:
            for p in glob.glob("hw_logs/**/*.jpg", recursive=True):
                add(z, p, man)
        z.writestr("MANIFEST.json", json.dumps(man, indent=2))
    mb = os.path.getsize(out) / 1e6
    print("packed %d files (%.1f MB) -> %s   upload this to Colab" % (len(man["files"]), mb, out))


def pack_for_jetson(a):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = "to_jetson_%s.zip" % stamp
    man = {"direction": "colab->jetson", "created": stamp, "files": {}, "training": []}
    for p in sorted(glob.glob("hw_logs/training/*/test_summary.json")):
        man["training"].append(json.load(open(p)))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob("deploy/*.npz") + glob.glob("deploy/time_model_*.json"):
            add(z, p, man)
        z.writestr("MANIFEST.json", json.dumps(man, indent=2))
    print("packed %d files -> %s   download this and copy it to the Jetson" % (len(man["files"]), out))


def unpack(a):
    with zipfile.ZipFile(a.zip) as z:
        man = json.loads(z.read("MANIFEST.json"))
        backup = os.path.join("deploy", "backup_%s" % time.strftime("%Y%m%d_%H%M%S"))
        for name, h in man["files"].items():
            data = z.read(name)
            if hashlib.sha256(data).hexdigest()[:16] != h:
                sys.exit("checksum mismatch for %s, aborting" % name)
            if os.path.exists(name):
                os.makedirs(backup, exist_ok=True)
                shutil.copy2(name, os.path.join(backup, os.path.basename(name)))
            os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
            with open(name, "wb") as f:
                f.write(data)
            print("  installed %s" % name)
    for t in man.get("training", []):
        print("  policy task=%s ik=%s seed=%s surveys=%s test=%s: mean objective %.3f"
              % (t.get("task"), t.get("ik"), t.get("seed"), t.get("surveys"), t.get("test_set"), t.get("policy_mean", float("nan"))))
    if os.path.isdir(backup):
        print("previous files backed up in %s" % backup)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack-for-colab", "pack-for-jetson", "unpack"])
    ap.add_argument("zip", nargs="?")
    ap.add_argument("--with-images", action="store_true", help="include per-cell images (large)")
    a = ap.parse_args()
    {"pack-for-colab": pack_for_colab, "pack-for-jetson": pack_for_jetson, "unpack": unpack}[a.cmd](a)
