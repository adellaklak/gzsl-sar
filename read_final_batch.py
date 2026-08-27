import re
from pathlib import Path

files = {
    "lb": "OAR.3018153.stdout", "vg_md": "OAR.3018154.stdout", "kd_md": "OAR.3018155.stdout",
    "mc_md": "OAR.3018156.stdout", "dirv2_bdavg": "OAR.3018088.stdout",
    "lb_dirv2_bdavg_md": "OAR.3018089.stdout", "dirv3_bdavg_md": "OAR.3018090.stdout",
    "lbac_dirv2_bdavg_md": "OAR.3018091.stdout", "ac_bdavg_md": "OAR.3018092.stdout",
    "mc_bdavg_md": "OAR.3018093.stdout",
}

print(f"{'config':25s} {'ZSL':>7s} {'S_Acc':>7s} {'U_Acc':>7s} {'H_Mean':>8s}")
results = []
for name, f in files.items():
    p = Path(f)
    if not p.exists():
        print(f"{name:25s} [MANQUANT: {f}]")
        continue
    txt = p.read_text()
    zsl = re.search(r"Best ZSL Acc:\s*([\d.]+)", txt)
    s = re.search(r"S_Acc:\s*([\d.]+)", txt)
    u = re.search(r"U_Acc:\s*([\d.]+)", txt)
    if not (zsl and s and u):
        print(f"{name:25s} [donnees incompletes, verifier stderr]")
        continue
    zsl_v, s_v, u_v = float(zsl.group(1)), float(s.group(1)), float(u.group(1))
    h = 2*s_v*u_v/(s_v+u_v) if (s_v+u_v) > 0 else 0
    results.append((name, zsl_v, s_v, u_v, h))
    print(f"{name:25s} {zsl_v:7.2f} {s_v:7.2f} {u_v:7.2f} {h:8.2f}")

print("\n=== Classement H_Mean ===")
for name, zsl_v, s_v, u_v, h in sorted(results, key=lambda x: -x[4]):
    print(f"{h:6.2f}  {name}")
