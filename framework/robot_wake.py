#!/usr/bin/env python3
"""After switching the robot off and on: check the serial port, power the servos, read the joints, and move SLOWLY,
in small steps, to the locked reference pose (deploy/locked_pose.json). Nothing is recalibrated or written.

  python3 robot_wake.py            # check + ask before moving
  python3 robot_wake.py --check    # only check (no motion)
"""
import json, math, os, subprocess, sys, time

PORT, BAUD = "/dev/ttyTHS1", 1000000
POSE_FILE = "deploy/locked_pose.json"
STEP_DEG, SPEED, TOL_DEG = 8.0, 15, 2.0


def port_free():
    try:
        r = subprocess.check_output(["fuser", PORT], stderr=subprocess.STDOUT).decode().strip()
        if r:
            print("PORT BUSY (%s): close live_align / other scripts, or  sudo kill -9 $(fuser %s 2>/dev/null)" % (r, PORT))
            return False
    except subprocess.CalledProcessError:
        pass
    except FileNotFoundError:
        pass
    return True


def read_angles(mc, tries=8):
    for _ in range(tries):
        try:
            q = mc.get_angles()
            if q and len(q) == 6 and all(isinstance(v, (int, float)) for v in q):
                return [float(v) for v in q]
        except Exception:
            pass
        time.sleep(0.4)
    return None


def wait_reached(mc, target, timeout=8.0):
    t0 = time.time(); q = None
    while time.time() - t0 < timeout:
        q = read_angles(mc, 2)
        if q and max(abs(a - b) for a, b in zip(q, target)) <= TOL_DEG:
            return q
        time.sleep(0.3)
    return q


def main():
    check_only = "--check" in sys.argv
    if not port_free():
        sys.exit(1)
    if not os.path.exists(POSE_FILE):
        sys.exit("no %s (run from ~/files/ins_new/mycobot_framework)" % POSE_FILE)
    target = [float(v) for v in json.load(open(POSE_FILE))["angles_deg"]]
    from pymycobot import MyCobot
    mc = MyCobot(PORT, BAUD)
    time.sleep(0.5)
    print("connected %s; powering the servos ..." % PORT)
    try:
        mc.power_on(); time.sleep(1.5)
    except Exception as e:
        print("power_on error: %s" % e)
    try:
        print("power on: %s | all servos enabled: %s" % (mc.is_power_on(), mc.is_all_servo_enable()))
    except Exception:
        pass
    q = read_angles(mc)
    if q is None:
        sys.exit("NO JOINT READING: check the robot's power supply and the USB/serial cable, wait 10 s after switching "
                 "on, then run again (if it persists, power-cycle the robot once more)")
    diff = [b - a for a, b in zip(q, target)]
    print("current : %s" % [round(v, 1) for v in q])
    print("locked  : %s" % [round(v, 1) for v in target])
    print("to move : %s deg (largest %.1f deg)" % ([round(v, 1) for v in diff], max(abs(v) for v in diff)))
    if check_only:
        return
    if input("move to the locked pose in small steps at speed %d? the workspace must be clear [y/N] " % SPEED).strip().lower() != "y":
        return
    n = max(1, int(math.ceil(max(abs(v) for v in diff) / STEP_DEG)))
    for i in range(1, n + 1):
        wp = [a + d * i / n for a, d in zip(q, diff)]
        mc.send_angles(wp, SPEED)
        got = wait_reached(mc, wp)
        print("  step %d/%d -> %s" % (i, n, "ok" if got and max(abs(a - b) for a, b in zip(got, wp)) <= TOL_DEG
                                          else "not reached (%s)" % (None if got is None else [round(v, 1) for v in got])))
    time.sleep(1.0)
    qf = read_angles(mc)
    err = [round(a - b, 2) for a, b in zip(qf, target)] if qf else None
    print("final   : %s" % ([round(v, 1) for v in qf] if qf else None))
    print("error   : %s deg -> %s" % (err, "AT THE LOCKED POSE" if err and max(abs(v) for v in err) <= TOL_DEG
                                       else "NOT at the locked pose: send me this output"))


if __name__ == "__main__":
    main()
