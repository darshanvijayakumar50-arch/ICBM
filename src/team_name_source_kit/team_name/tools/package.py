#!/usr/bin/env python3
"""
Build the submission zip.

usage (run from the folder that CONTAINS your team folder, or pass paths):
    python package.py <team_folder> [--run <input imu csv|zip|folder>]

--run : runs src/main.py on the given input first, writing estimated_trajectory.txt and metadata.json
        into the team folder, validates the result, then zips.
Result: <team_folder>.zip containing <team_folder>/{estimated_trajectory.txt, metadata.json, src/, README.md}
"""
import os
import subprocess
import sys
import zipfile


def main():
    team = os.path.abspath(sys.argv[1]).rstrip("/\\")
    name = os.path.basename(team)
    if "--run" in sys.argv:
        inp = sys.argv[sys.argv.index("--run") + 1]
        out = os.path.join(team, "estimated_trajectory.txt")
        meta = os.path.join(team, "metadata.json")
        r = subprocess.run([sys.executable, os.path.join(team, "src", "main.py"), inp, "-o", out, "--meta", meta])
        if r.returncode:
            print("main.py failed; not packaging"); return 1
        v = subprocess.run([sys.executable, os.path.join(team, "tools", "validate_submission.py"), inp, out]) \
            if not inp.lower().endswith(".zip") and os.path.isfile(inp) else None
        if v is not None and v.returncode:
            print("validation failed; not packaging"); return 1
    must = ["estimated_trajectory.txt", os.path.join("src", "main.py"), "README.md"]
    for m in must:
        if not os.path.isfile(os.path.join(team, m)):
            print("MISSING required file:", m); return 1
    zpath = team + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ["estimated_trajectory.txt", "metadata.json", "README.md", os.path.join("src", "main.py")]:
            full = os.path.join(team, rel)
            if os.path.isfile(full):
                z.write(full, name + "/" + rel.replace("\\", "/"))
        tools = os.path.join(team, "tools")
        if os.path.isdir(tools):
            for f in sorted(os.listdir(tools)):
                if f.endswith(".py"):
                    z.write(os.path.join(tools, f), name + "/tools/" + f)
    print("created", zpath)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
