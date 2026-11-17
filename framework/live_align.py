#!/usr/bin/env python3
"""Live camera + step buttons for alignment AND corner teaching.
ALIGNMENT OVERLAY: big green cross = image centre (10 px green ring = centred, 25 px orange ring = acceptable);
green crosses = marker centres (with ids); yellow cross = midpoint of the two markers; magenta cross = drawn-box
centre; the arrow shows where the target must go. Jog until the bottom line says CENTRED and markers 2/2 OK,
then press "Lock pose" (the offset is saved in locked_pose.json). Add --chess to also draw a chessboard.
+ = anti-clockwise, - = clockwise.
Buttons:
  per joint: -10 -5 -1 [current] +1 +5 +10 [set]
  Sync, Home, Capture, Lock pose, Return, Quit
  Save C1..C4  → save current pose to deploy/corner_N.json
Servos stay powered throughout."""
import sys, os, time, json, subprocess, threading
import numpy as np, cv2, tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

try:
    r = subprocess.check_output(["fuser", "/dev/ttyTHS1"], stderr=subprocess.STDOUT).decode()
    if r.strip():
        print(f"PORT BUSY: {r.strip()}")
        print("Run:  sudo kill -9 $(fuser /dev/ttyTHS1 2>/dev/null)")
        sys.exit(1)
except subprocess.CalledProcessError:
    pass

from pymycobot import MyCobot
PORT, BAUD = "/dev/ttyTHS1", 1000000
POSE_FILE = "deploy/locked_pose.json"
SNAP_DIR = "hw_logs"
PX_PER_MM = 2.25          # approximate image scale at the reference height (only for the mm hint on screen)
SHOW_CHESS = "--chess" in sys.argv
ALIGN = {"offset_px": None, "target": None}
os.makedirs(SNAP_DIR, exist_ok=True); os.makedirs("deploy", exist_ok=True)

print(f"Connecting {PORT} ...")
robot = MyCobot(PORT, BAUD)
robot.power_on(); time.sleep(0.5)
try:
    start_angles = [float(x) for x in robot.get_angles()]
except Exception:
    start_angles = [0, 0, 0, 0, 0, 0]
print("start angles:", [round(x, 2) for x in start_angles])

# ---------------- camera ----------------
class Cam:
    def __init__(self, idx=0):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened(): raise RuntimeError("camera 0 failed")
        self.latest = None; self.flag = True
        threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        for _ in range(10): self.cap.read()
        while self.flag:
            ok, f = self.cap.read()
            if ok: self.latest = f
            else:  time.sleep(0.05)
    def stop(self):
        self.flag = False; time.sleep(0.1); self.cap.release()

cam = Cam(0)

if hasattr(cv2.aruco, "ArucoDetector"):
    _d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _det = cv2.aruco.ArucoDetector(_d, cv2.aruco.DetectorParameters())
    def detect_aruco(g): c, i, _ = _det.detectMarkers(g); return c, i
else:
    _d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    _p = cv2.aruco.DetectorParameters_create()
    def detect_aruco(g): c, i, _ = cv2.aruco.detectMarkers(g, _d, parameters=_p); return c, i

# ---------------- alignment overlay ----------------
# green cross   = image centre (where the box centre must go); rings = 10 px (good) and 25 px (acceptable)
# green crosses = centre of every detected marker (with its id)
# yellow cross  = midpoint of the two markers (they sit at opposite corners of the drawn box)
# magenta cross = centre of the drawn box (when the box outline is found)
# arrow         = from the target to the image centre: move the arm so the target follows the arrow
_BOX = {"n": 0, "box": None}


def _find_box(img):
    """Drawn-box corners via the framework's detector (refreshed every few frames; None if unavailable)."""
    _BOX["n"] += 1
    if _BOX["n"] % 8 == 1:
        try:
            import board as BD
            from measure_board import find_box
            dets = BD.detect_markers(img)
            _BOX["box"] = find_box(img, dets if len(dets) >= 4 else None)
        except Exception:
            _BOX["box"] = None
    return _BOX["box"]


def _cross(f, p, col, size=18, thick=2):
    p = (int(round(p[0])), int(round(p[1])))
    cv2.drawMarker(f, p, col, cv2.MARKER_CROSS, size, thick, cv2.LINE_AA)


def annotate(frame):
    f = frame.copy()
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    chess, corners = cv2.findChessboardCorners(g, (7, 6), None) if SHOW_CHESS else (False, None)
    if chess: cv2.drawChessboardCorners(f, (7, 6), corners, chess)
    c, i = detect_aruco(g)
    n_aruco = 0
    centres = []
    if i is not None:
        n_aruco = len(i)
        for cc, ii in zip(c, i.flatten()):
            pts = cc.reshape(-1, 2)
            cv2.polylines(f, [pts.astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
            m = pts.mean(0); centres.append(m)
            _cross(f, m, (0, 255, 0), 22, 2)
            cv2.putText(f, "id %d" % ii, (int(m[0]) + 8, int(m[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
    uc = np.array([w / 2.0, h / 2.0])
    # image centre: big green cross + tolerance rings
    _cross(f, uc, (0, 255, 0), 40, 2)
    cv2.circle(f, (int(uc[0]), int(uc[1])), 10, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(f, (int(uc[0]), int(uc[1])), 25, (0, 200, 255), 1, cv2.LINE_AA)
    target, how = None, ""
    box = _find_box(frame)
    if box is not None:
        b = np.asarray(box, float).reshape(-1, 2)
        cv2.polylines(f, [b.astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
        target, how = b.mean(0), "box centre"
        _cross(f, target, (255, 0, 255), 26, 2)
    if len(centres) >= 2:
        mid = np.mean(centres[:2], 0)
        cv2.line(f, tuple(int(v) for v in centres[0]), tuple(int(v) for v in centres[1]), (0, 220, 220), 1, cv2.LINE_AA)
        _cross(f, mid, (0, 255, 255), 26, 2)
        if target is None:
            target, how = mid, "marker midpoint"
    if target is not None:
        d = target - uc; e = float(np.hypot(*d))
        col = (0, 255, 0) if e <= 10 else ((0, 200, 255) if e <= 25 else (0, 0, 255))
        cv2.arrowedLine(f, (int(target[0]), int(target[1])), (int(uc[0]), int(uc[1])), col, 2, cv2.LINE_AA, tipLength=0.15)
        side = []
        if abs(d[0]) > 5: side.append("LEFT" if d[0] < 0 else "RIGHT")
        if abs(d[1]) > 5: side.append("UP" if d[1] < 0 else "DOWN")
        txt = "%s %+.0f,%+.0f px (%.0f px ~ %.0f mm) %s" % (how, d[0], d[1], e, e / PX_PER_MM,
                                                          "CENTRED" if e <= 10 else "target is " + "/".join(side) + " of centre")
        ALIGN["offset_px"] = [float(d[0]), float(d[1])]; ALIGN["target"] = how
    else:
        txt = "no target: need both markers or the drawn box in view"; col = (0, 0, 255)
        ALIGN["offset_px"] = None; ALIGN["target"] = None
    cv2.rectangle(f, (0, h - 46), (w, h), (0, 0, 0), -1)
    cv2.putText(f, txt, (8, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    mk = "markers %d/2 %s" % (n_aruco, "OK" if n_aruco >= 2 else "- BOTH MARKERS MUST BE FULLY VISIBLE")
    cv2.putText(f, mk, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if n_aruco >= 2 else (0, 0, 255), 1, cv2.LINE_AA)
    bright = float(g.mean())
    return f, chess, n_aruco, bright


# ---------------- GUI ----------------
class App:
    def __init__(self, root):
        self.root = root; root.title("Live align + corner teach")
        self.angles = start_angles[:]
        self.vars = [tk.StringVar(value=f"{a:.1f}") for a in self.angles]
        self.busy = False
        self.last_send_ms = 0.0

        # LEFT: camera
        left = tk.Frame(root); left.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.img_label = tk.Label(left, bg="black", width=640, height=480)
        self.img_label.pack()
        self.detect_label = tk.Label(left, text="detector: --", anchor="w",
                                      font=("Mono", 10), fg="gray")
        self.detect_label.pack(fill="x", pady=(4, 0))
        self.status = tk.Label(left, text="ready", anchor="w", fg="darkgreen",
                                font=("Mono", 10), wraplength=640, justify="left")
        self.status.pack(fill="x", pady=(2, 0))

        # RIGHT: controls
        right = tk.Frame(root); right.grid(row=0, column=1, padx=6, pady=6, sticky="n")
        tk.Label(right, text="Align + teach corners", font=("Sans", 11, "bold")
                 ).grid(row=0, column=0, columnspan=9, pady=(0, 4))
        tk.Label(right, text="+ = anti-clockwise,  − = clockwise",
                 fg="gray", font=("Sans", 9)).grid(row=1, column=0, columnspan=9, pady=(0, 8))

        # header
        tk.Label(right, text="Joint", width=6).grid(row=2, column=0)
        for j, t in enumerate(["-10","-5","-1","current","+1","+5","+10"]):
            tk.Label(right, text=t, width=6).grid(row=2, column=1+j)

        for j in range(6):
            r = 3 + j
            tk.Label(right, text=f"J{j+1}", font=("Sans", 10, "bold")).grid(row=r, column=0)
            for k, mag in enumerate([10, 5, 1]):
                tk.Button(right, text=f"-{mag}", width=5,
                          command=lambda jj=j, m=mag: self.step(jj, -m)
                          ).grid(row=r, column=1+k, padx=1, pady=1)
            tk.Entry(right, textvariable=self.vars[j], width=8, justify="center",
                     font=("Mono", 10)).grid(row=r, column=4, padx=2, pady=1)
            for k, mag in enumerate([1, 5, 10]):
                tk.Button(right, text=f"+{mag}", width=5,
                          command=lambda jj=j, m=mag: self.step(jj, +m)
                          ).grid(row=r, column=5+k, padx=1, pady=1)
            tk.Button(right, text="set", width=5,
                      command=lambda jj=j: self.set_joint(jj)
                      ).grid(row=r, column=8, padx=(6, 0), pady=1)

        # speed
        rf = tk.Frame(right); rf.grid(row=9, column=0, columnspan=9, pady=(10, 0), sticky="we")
        tk.Label(rf, text="Speed").pack(side="left")
        self.speed = tk.Scale(rf, from_=5, to=40, orient="horizontal", length=240, resolution=1)
        self.speed.set(20); self.speed.pack(side="left", padx=6)
        self.speed_lbl = tk.Label(rf, text="20", width=4); self.speed_lbl.pack(side="left")
        self.speed.config(command=lambda v: self.speed_lbl.config(text=v))

        # action buttons
        bf = tk.Frame(right); bf.grid(row=10, column=0, columnspan=9, pady=(10, 0), sticky="we")
        for label, fn in [("Sync", self.sync), ("Home", self.home),
                          ("Capture", self.capture), ("Lock pose", self.lock),
                          ("Return", self.return_pose), ("Quit", self.quit)]:
            tk.Button(bf, text=label, command=fn).pack(side="left", expand=True, fill="x", padx=2)

        # corner save buttons
        cf = tk.Frame(right); cf.grid(row=11, column=0, columnspan=9, pady=(6, 0), sticky="we")
        tk.Label(cf, text="Teach corner:", font=("Sans", 9)).pack(side="left", padx=(0, 4))
        for n in [1, 2, 3, 4]:
            tk.Button(cf, text=f"Save C{n}",
                      bg="#e8f4ff",
                      command=lambda nn=n: self.save_corner(nn)
                      ).pack(side="left", expand=True, fill="x", padx=2)

        # IK preview
        ikf = tk.Frame(right); ikf.grid(row=12, column=0, columnspan=9, pady=(8, 0), sticky="we")
        tk.Label(ikf, text="IK preview:").pack(side="left")
        self.ik_x = tk.Entry(ikf, width=6); self.ik_x.insert(0, "250"); self.ik_x.pack(side="left", padx=1)
        self.ik_y = tk.Entry(ikf, width=6); self.ik_y.insert(0, "0"); self.ik_y.pack(side="left", padx=1)
        self.ik_z = tk.Entry(ikf, width=6); self.ik_z.insert(0, "120"); self.ik_z.pack(side="left", padx=1)
        tk.Button(ikf, text="compute", command=self.ik_preview).pack(side="left", padx=6)

        self.frame_info = tk.Label(right, text="", anchor="w",
                                    font=("Mono", 9), fg="gray")
        self.frame_info.grid(row=13, column=0, columnspan=9, sticky="we", pady=(6, 0))

        self.root.after(30, self.tick)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    # ---- joint commands ----
    def _send(self, new_angles, msg):
        if self.busy:
            self.status.config(text="busy (waiting)", fg="orange"); return
        self.busy = True
        try:
            t0 = time.perf_counter()
            robot.send_angles([float(x) for x in new_angles], int(self.speed.get()))
            self.last_send_ms = (time.perf_counter() - t0) * 1000
            self.angles = list(new_angles)
            for j in range(6):
                self.vars[j].set(f"{self.angles[j]:.1f}")
            self.status.config(text=f"{msg}    send={self.last_send_ms:.0f}ms    "
                                    f"angles={[round(a,1) for a in self.angles]}",
                                fg="darkgreen")
        except Exception as e:
            self.status.config(text=f"send error: {e}", fg="red")
        finally:
            self.busy = False

    def step(self, j, delta):
        new = list(self.angles)
        new[j] = float(np.clip(new[j] + delta, -168, 168))
        self._send(new, f"J{j+1} {delta:+d} → {new[j]:.1f}")

    def set_joint(self, j):
        try: val = float(self.vars[j].get())
        except ValueError:
            self.status.config(text=f"invalid J{j+1}", fg="red"); return
        val = float(np.clip(val, -168, 168))
        new = list(self.angles); new[j] = val
        self._send(new, f"J{j+1} := {val:.1f}")

    # ---- actions ----
    def sync(self):
        if self.busy: return
        self.busy = True
        try:
            q = [float(x) for x in robot.get_angles()]
            if q and len(q) == 6:
                self.angles = q
                for j in range(6): self.vars[j].set(f"{q[j]:.1f}")
                co = robot.get_coords()
                self.status.config(text=f"synced  angles={[round(a,1) for a in q]}  "
                                        f"coords={[round(float(x),1) for x in co]}",
                                    fg="blue")
        except Exception as e:
            self.status.config(text=f"sync error: {e}", fg="red")
        finally: self.busy = False

    def home(self):
        self._send([-80.15, -0.87, -3.69, 1.14, -2.37, 146.33], "HOME")

    def capture(self):
        if cam.latest is None: return
        p = os.path.join(SNAP_DIR, f"align_{int(time.time())}.jpg")
        cv2.imwrite(p, cam.latest)
        self.status.config(text=f"capture → {p}", fg="blue")

    def lock(self):
        try:
            q = [float(x) for x in robot.get_angles()]
            co = robot.get_coords()
            _, chess, aruco, bright = annotate(cam.latest if cam.latest is not None
                                               else np.zeros((480,640,3), np.uint8))
            payload = {"angles_deg": q, "coords_mm": list(co),
                       "detected": {"chessboard": bool(chess), "aruco": bool(aruco),
                                    "brightness": bright},
                       "align_offset_px": ALIGN["offset_px"], "align_target": ALIGN["target"],
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(POSE_FILE, "w") as fp:
                json.dump(payload, fp, indent=2)
            self.angles = q
            for j in range(6): self.vars[j].set(f"{q[j]:.1f}")
            off = ALIGN["offset_px"]
            note = ("  target offset %+.0f,%+.0f px" % (off[0], off[1])) if off else "  (no target visible!)"
            self.status.config(text=f"LOCKED → {POSE_FILE}  coords={[round(float(x),1) for x in co]}{note}",
                                fg="blue" if off and (off[0] ** 2 + off[1] ** 2) ** 0.5 <= 25 else "orange")
        except Exception as e:
            self.status.config(text=f"lock error: {e}", fg="red")

    def save_corner(self, n):
        try:
            q = [float(x) for x in robot.get_angles()]
            co = robot.get_coords()
            path = f"deploy/corner_{n}.json"
            with open(path, "w") as fp:
                json.dump({"angles_deg": q, "coords_mm": list(co),
                           "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, indent=2)
            self.status.config(text=f"CORNER {n} SAVED  angles={[round(float(x),1) for x in q]}  "
                                    f"coords={[round(float(x),1) for x in co]}",
                                fg="blue")
        except Exception as e:
            self.status.config(text=f"save corner {n} error: {e}", fg="red")

    def return_pose(self):
        try:
            d = json.load(open(POSE_FILE))
        except Exception as e:
            self.status.config(text=f"return: {e}", fg="red"); return
        self._send(d["angles_deg"], "return → locked pose")

    def ik_preview(self):
        try:
            tx = float(self.ik_x.get()) / 1000.0
            ty = float(self.ik_y.get()) / 1000.0
            tz = float(self.ik_z.get()) / 1000.0
        except ValueError:
            self.status.config(text="ik preview: invalid xyz", fg="red"); return
        try:
            from mycobot_metrology.ik import ik
            q = np.deg2rad(self.angles)
            q_tgt, err = ik([tx, ty, tz], q)
            self.status.config(text=f"IK ({tx*1000:.0f},{ty*1000:.0f},{tz*1000:.0f})mm "
                                    f"→ {[round(np.rad2deg(a),1) for a in q_tgt]}  "
                                    f"err={err*1000:.1f}mm",
                                fg="darkgreen" if err < 0.01 else "orange")
        except Exception as e:
            self.status.config(text=f"ik error: {e}", fg="red")

    # ---- periodic ----
    def tick(self):
        f = cam.latest
        if f is not None:
            disp, chess, aruco, bright = annotate(f)
            off = ALIGN["offset_px"]
            self.detect_label.config(text=f"brightness={bright:.0f}   aruco={aruco}/2   "
                                          + (f"{ALIGN['target']} offset {off[0]:+.0f},{off[1]:+.0f} px"
                                             if off else "no target"),
                                     fg=("darkgreen" if off and (off[0] ** 2 + off[1] ** 2) ** 0.5 <= 10 else
                                         "orange" if off and (off[0] ** 2 + off[1] ** 2) ** 0.5 <= 25 else "red"))
            rgb = cv2.cvtColor(cv2.resize(disp, (640, 480)), cv2.COLOR_BGR2RGB)
            imtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.img_label.imgtk = imtk
            self.img_label.config(image=imtk)
        self.frame_info.config(text=f"last send: {self.last_send_ms:.0f} ms   "
                                    f"busy={self.busy}")
        self.root.after(30, self.tick)

    def quit(self):
        try: cam.stop()
        except: pass
        try: robot.close()
        except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
