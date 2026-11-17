"""myCobot 280 forward kinematics (DH). Two tables are provided:
  "repo": your original kinematics.py (d5 = 45.6 mm, d6 = 0)
  "alt" : d5 = 75.05 mm, d6 = 45.6 mm, joint-5 offset +90 deg (published variant)
Select with config.KIN_MODEL. Your locked pose (firmware coords -9.6, -202.6, 208.4) matches "alt"
exactly in x, y and orientation, with a constant +8.2 mm z offset (config.KIN_Z_OFFSET_MM);
the "repo" table is ~51 mm off. fk_check.py verifies this on 13 poses."""
import numpy as np
try:
    from config import KIN_MODEL, KIN_Z_OFFSET_MM
except Exception:
    KIN_MODEL, KIN_Z_OFFSET_MM = "alt", 8.2

_TABLES = {
    "repo": dict(D=np.array([131.22, 0.0, 0.0, 63.4, 45.6, 0.0]) / 1000.0,
                 A=np.array([0.0, -110.4, -96.0, 0.0, 0.0, 0.0]) / 1000.0,
                 AL=np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]),
                 OFF=np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])),
    "alt":  dict(D=np.array([131.22, 0.0, 0.0, 63.4, 75.05, 45.6]) / 1000.0,
                 A=np.array([0.0, -110.4, -96.0, 0.0, 0.0, 0.0]) / 1000.0,
                 AL=np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]),
                 OFF=np.array([0.0, -np.pi/2, 0.0, -np.pi/2, np.pi/2, 0.0])),
}
LIM = np.deg2rad(np.array([168, 168, 168, 168, 168, 168]))


def _dh(t, d, a, al):
    ct, st, ca, sa = np.cos(t), np.sin(t), np.cos(al), np.sin(al)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0.,     sa,     ca,    d],
                     [0.,     0.,     0.,   1.]])


def fk(q, model=None, z_offset_mm=None):
    """Flange pose in the firmware's base frame (metres)."""
    m = model or KIN_MODEL
    P = _TABLES[m]
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(q[i] + P["OFF"][i], P["D"][i], P["A"][i], P["AL"][i])
    if m == KIN_MODEL or z_offset_mm is not None:
        T[2, 3] += (KIN_Z_OFFSET_MM if z_offset_mm is None else z_offset_mm) / 1000.0
    return T


def clamp_joints(q):
    return np.clip(q, -LIM, LIM)
