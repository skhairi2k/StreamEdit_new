#!/bin/bash
# R50 step jz-overlay (ON THE JEAN ZAY FRONTAL: PyPI and GitHub must be reachable; compute nodes have no internet).
# Build the Python the Wan-side jobs run in: the IDRIS H100 module ($JZ_MODULE, torch 2.5.1 like our five-bench env) PLUS a pinned pip
# overlay in $WORK/pyenv/five-bench, so no conda env is unpacked ($WORK is limited to 500k files; the module costs none).
#
#   1. load `arch/h100` + the module; refuse python < 3.10 (transformers 5.x)
#   2. plan: compare the module's packages with ours (stage freeze/five-bench.txt = `pip list --format=freeze` of the lab env). Skipped: torch,
#      torchvision, torchaudio, triton, nvidia-*, pip, setuptools, wheel (the module's own builds). Every other package that is missing or at
#      another version is pinned to OUR version.
#      Also left to the module, on purpose: numpy (the module's torch is compiled against its NumPy 1.x; ours, 2.x, makes torch.from_numpy /
#      .numpy() fail) and OpenCV (4.12+ needs numpy>=2; the module's is kept, else 4.10.0.84). Added although the lab env lacks it: peft 0.18.1
#      when the module's peft is < 0.17 (diffusers 0.36 refuses an old peft), mistral-common 1.8.8 when the module's is older (transformers 5.3). Anything of that kind found in the overlay is removed first.
#   3. `pip install --user --no-deps --ignore-installed --no-cache-dir -r ...` into PYTHONUSERBASE=$WORK/pyenv/five-bench (the user site comes
#      before the module's site-packages, so e.g. our transformers 5.x shadows the module's 4.x). A failed bulk install is retried one package at
#      a time to name the culprits.
#   4. flash-attn (Wan-Edit asserts it; the FiVE-Acc Qwen model asks for it): kept if the module already ships our version, else the prebuilt
#      wheel matching the module's torch / CUDA / python / C++ ABI is taken from $WORK/pyenv/wheels or downloaded from the flash-attention
#      GitHub release (never compiled on the frontal). If there is no such wheel: download it on a machine with access, copy it to
#      $WORK/pyenv/wheels/, rerun.
#   5. write the interpreter wrapper $WORK/pyenv/five-bench/python (module purge + load, PYTHONPATH = overlay first + the pyiqa source
#      $WORK/code/iqa_cli, HF_HOME / TORCH_HOME = $WORK/cache/{huggingface,torch}, exec python). Job scripts call this wrapper (jz/r50_port.sh).
#   6. verify THROUGH THE WRAPPER: every pin that was meant to be applied is, everything we installed is imported from the overlay (not shadowed
#      by the module), a real torch<->numpy round trip works without warnings, numpy / scipy / pandas / cv2 / PIL / torchvision import, `pip check`
#      has no complaint made BY a package we installed (the module's own packages complaining about our newer pins are only counted; full list in
#      overlay_pipcheck.txt), key modules import from the overlay.
#   Not installed: cotracker (unused by R49; no motion-fidelity metric), editable installs (pyiqa is reached through PYTHONPATH instead).
#
# Usage: bash jz/r50_pip_overlay.sh              plan + install + wrapper + verify (idempotent: a rerun installs only what still differs)
#        bash jz/r50_pip_overlay.sh --dry-run    plan only, nothing is installed or written
# R50_ENV=<file> overrides jz/r50_jz.env; R50_FREEZE=<file> overrides the package list (tests).
# Ends with "[r50_overlay] installed N, mismatches M, files F" and "[r50_overlay] failures: K" (0 = go on to jz-port).
set -u

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
say() { echo "[r50_overlay] $*"; }
[ -f "$CFG" ] || { say "FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep -q '<'; then say "FATAL: unfilled <...> value in $CFG"; exit 1; fi
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
MODULE=${JZ_MODULE:?JZ_MODULE not set in $CFG}
[ -d "$WORK" ] || { say "FATAL: $WORK does not exist on $(hostname) -- this script is for Jean Zay only"; exit 1; }
FREEZE=${R50_FREEZE:-$WORK/r50_stage/freeze/five-bench.txt}
[ -s "$FREEZE" ] || { say "FATAL: $FREEZE missing -- run jz/r50_setup_dirs.sh (the stage must have been pushed)"; exit 1; }
ENVDIR=$WORK/pyenv/five-bench
WRAP=$ENVDIR/python
TOOLS=$WORK/pyenv/r50_overlay_tools.py
TMPD=$WORK/pyenv/tmp
WHEELS=$WORK/pyenv/wheels
IQA=$WORK/code/iqa_cli
FAILED=0

# ---- 1. the module
say "1. module: arch/h100 + ${MODULE} on $(hostname) $(date '+%F %T')"
if ! type module >/dev/null 2>&1; then
  for F in /usr/share/Modules/init/bash /etc/profile.d/modules.sh "${MODULESHOME:-/nonexistent}/init/bash"; do [ -f "$F" ] && { . "$F"; break; }; done
fi
type module >/dev/null 2>&1 || { say "FATAL: no 'module' command in this shell (log in to the frontal with a login shell)"; exit 1; }
module purge >/dev/null 2>&1
module load arch/h100 "$MODULE" >/dev/null 2>&1 || { say "FATAL: 'module load arch/h100 ${MODULE}' failed -- module avail pytorch-gpu"; exit 1; }
command -v python >/dev/null || { say "FATAL: no python after loading the module"; exit 1; }
PYV=$(python -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
say "   python ${PYV} ($(command -v python)), pip $(python -m pip --version 2>/dev/null | awk '{print $2}')"
python -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { say "FATAL: python ${PYV} < 3.10: transformers 5.x (our pin) cannot be installed; pick a newer module"; exit 1; }
python - <<'PYEOF' || { say "FATAL: the module's python cannot import torch"; exit 1; }
import torch
print("[r50_overlay]    torch %s (cuda build %s, cxx11 abi %s)" % (torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI))
PYEOF

export PYTHONUSERBASE=$ENVDIR
export TMPDIR=$TMPD
mkdir -p "$ENVDIR" "$TMPD" "$WHEELS"
USER_SITE=$(python -c 'import site; print(site.getusersitepackages())')
mkdir -p "$USER_SITE"
export PYTHONPATH="$USER_SITE${PYTHONPATH:+:$PYTHONPATH}"          # the overlay wins over anything the module puts on PYTHONPATH
say "   overlay: ${USER_SITE}"

# ---- the helper (plan / wheel-url / verify), kept next to the overlay so the wrapper-based verify can run it too
cat > "$TOOLS" <<'PYEOF'
"""r50_pip_overlay helper. plan FREEZE REQ PLAN | wheel VERSION | verify FREEZE PLAN USER_SITE"""
import importlib.metadata as md
import json
import re
import subprocess
import sys
from pathlib import Path

SKIP = {"torch", "torchvision", "torchaudio", "triton", "pip", "setuptools", "wheel"}
OPENCV = ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless")
OPENCV_NP1 = "4.10.0.84"          # last OpenCV that runs on NumPy 1.26 (4.12+ declares numpy>=2); only used if the module has no OpenCV at all
PEFT_FIX = ("0.17.0", "0.18.1")   # diffusers 0.36 refuses peft < 0.17; peft 0.17.x does not import with transformers 5.3; 0.18.1 is the oldest that works
MISTRAL_FIX = "1.8.8"            # transformers 5.3 (dependency_versions_table) wants mistral-common >= 1.8.8; the module's older one breaks every AutoProcessor load
MISTRAL_DEPS = {"pydantic-extra-types": "2.10.5", "pycountry": "24.6.1"}   # mistral-common 1.8.8 imports pydantic_extra_types.language_code (needs pycountry); its other requirements are in the module
KEY_MODULES = {"transformers": "transformers", "diffusers": "diffusers", "huggingface-hub": "huggingface_hub", "tokenizers": "tokenizers",
               "safetensors": "safetensors", "accelerate": "accelerate", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
               "timm": "timm", "torchmetrics": "torchmetrics", "lpips": "lpips", "omegaconf": "omegaconf", "einops": "einops",
               "opencv-python": "cv2", "qwen-vl-utils": "qwen_vl_utils", "flash-attn": "flash_attn", "pillow": "PIL", "pyiqa": "pyiqa", "peft": "peft", "mistral-common": "mistral_common", "pydantic-extra-types": "pydantic_extra_types", "pycountry": "pycountry"}


def norm(n):
    return re.sub(r"[-_.]+", "-", n.strip()).lower()


def skipped(n):
    return n in SKIP or n.startswith("nvidia-")


def read_freeze(path):
    pins, unpinned = {}, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line)
        if m:
            pins[norm(m.group(1))] = m.group(2)
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*@", line)
        if m:
            unpinned.append(norm(m.group(1)))
        else:
            print(f"[r50_overlay] WARNING unparsed freeze line: {line}")
    return pins, unpinned


def installed(exclude=None):
    """name -> (version, location): the first distribution found on sys.path, i.e. the one `import` would use.
    With `exclude` (a path prefix such as the overlay's site-packages) those distributions are ignored: the module alone."""
    out = {}
    for d in md.distributions():
        name = d.metadata["Name"]
        loc = str(getattr(d, "_path", ""))
        if name and norm(name) not in out and not (exclude and loc.startswith(exclude)):
            out[norm(name)] = (d.version, loc)
    return out


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def cmd_plan(freeze, req_out, plan_out, user_site):
    pins, unpinned = read_freeze(freeze)
    have = installed()                          # what a job imports: overlay first
    base = installed(exclude=user_site)         # the module alone
    notes, left = [], set()                     # `left`: pins deliberately NOT applied
    # NumPy: the module's torch is compiled against ITS NumPy; another major version breaks torch <-> numpy
    # ("Failed to initialize NumPy": from_numpy / .numpy() raise). So numpy stays the module's.
    mod_np = base.get("numpy", (None,))[0]
    if "numpy" in pins:
        left.add("numpy")
        notes.append(f"numpy: ours {pins['numpy']} NOT applied, the module's {mod_np or 'absent'} stays (its torch is compiled against it)")
    # OpenCV 4.12+ declares numpy>=2: with the module's NumPy 1.x use the module's OpenCV, else the last release that runs on NumPy 1.26
    cv = [n for n in OPENCV if n in pins]
    if cv and mod_np and vkey(mod_np)[0] < 2:
        if any(n in base for n in OPENCV):
            left |= set(cv)
            notes.append(f"opencv: ours {pins[cv[0]]} (needs numpy>=2) NOT applied, the module's {[n + ' ' + base[n][0] for n in OPENCV if n in base]} stays")
        else:
            keep = "opencv-python-headless" if "opencv-python-headless" in cv else cv[0]
            left |= set(cv) - {keep}
            pins[keep] = OPENCV_NP1
            notes.append(f"opencv: the module has none; {keep}=={OPENCV_NP1} instead of ours (needs numpy>=2)")
    # peft: diffusers refuses the module's old peft; our lab env has none, so it is not in the freeze
    pv = base.get("peft", (None,))[0]
    if pv and vkey(pv) < vkey(PEFT_FIX[0]) and "peft" not in pins:
        pins["peft"] = PEFT_FIX[1]
        notes.append(f"peft: the module's {pv} is older than diffusers allows (>={PEFT_FIX[0]}): peft=={PEFT_FIX[1]} added to the overlay")
    # mistral-common: transformers 5.3 imports it as soon as it is installed (our lab env has none, so it is not in the freeze) and needs >= 1.8.8
    mv = base.get("mistral-common", (None,))[0]
    if mv and vkey(mv) < vkey(MISTRAL_FIX) and "mistral-common" not in pins:
        pins["mistral-common"] = MISTRAL_FIX
        for dep, ver in MISTRAL_DEPS.items():             # --no-deps: the requirements of mistral-common 1.8.8 that the module lacks
            if dep not in base and dep not in pins:
                pins[dep] = ver
                notes.append(f"{dep}=={ver} added (required by mistral-common {MISTRAL_FIX}; the module has none)")
        notes.append(f"mistral-common: the module's {mv} is older than transformers 5.3 allows (>={MISTRAL_FIX}): mistral-common=={MISTRAL_FIX} added to the overlay")
    install, same, wheel, skipped_n, effective = [], 0, None, 0, {}
    for n, v in sorted(pins.items()):
        if skipped(n) or n in left:
            skipped_n += 1
            continue
        effective[n] = v
        cur = have.get(n, (None,))[0]
        if cur == v:
            same += 1
        elif n == "flash-attn":
            wheel = {"version": v, "have": cur}
        else:
            install.append({"name": n, "ours": v, "module": cur})
    # packages that must NOT come from the overlay but are there (e.g. the numpy 2.x of an earlier run)
    purge = sorted(n for n in have if (skipped(n) or n in left) and have[n][1].startswith(user_site))
    notes += [f"{n}: pinned only by a local path in the lab env; module copy {'kept' if n in have else 'ABSENT (not installable here)'}" for n in unpinned]
    Path(req_out).write_text("".join(f"{i['name']}=={i['ours']}\n" for i in install))
    Path(plan_out).write_text(json.dumps({"install": install, "wheel": wheel, "same": same, "skipped": skipped_n, "unpinned": unpinned,
                                          "effective": effective, "left": sorted(left), "purge": purge, "notes": notes}, indent=1))
    print(f"[r50_overlay]    ours: {len(pins)} pins ({skipped_n} left to the module: torch family, nvidia-*, pip, numpy, opencv...), identical in the module: {same}, "
          f"to install: {len(install)}, flash-attn: {'wheel ' + wheel['version'] + ' (module has ' + str(wheel['have']) + ')' if wheel else 'kept'}")
    if purge:
        print(f"[r50_overlay]    to REMOVE from the overlay (left to the module): {', '.join(purge)}")
    for i in install[:12]:
        print(f"[r50_overlay]      {i['name']:<26} ours {i['ours']:<14} module {i['module'] or 'absent'}")
    if len(install) > 12:
        print(f"[r50_overlay]      ... {len(install) - 12} more (see overlay.txt)")
    for t in notes:
        print(f"[r50_overlay]    NOTE {t}")


def cmd_wheel(version):
    import torch
    cu = "cu12" if str(torch.version.cuda).startswith("12") else "cu118"
    tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    py = "cp%d%d" % sys.version_info[:2]
    name = f"flash_attn-{version}+{cu}torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
    print(name)
    print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{name}")


def cmd_verify(plan_path, user_site):
    plan = json.loads(Path(plan_path).read_text())
    eff = plan["effective"]                     # the pins that were meant to be applied (numpy / opencv left out on purpose)
    have = installed()
    ours = {i["name"] for i in plan["install"]} | ({"flash-attn"} if plan["wheel"] else set())
    problems = 0
    mism = [(n, v, have.get(n, (None,))[0]) for n, v in sorted(eff.items()) if have.get(n, (None,))[0] != v]
    for n, v, h in mism:
        print(f"[r50_overlay]    MISMATCH {n}: ours {v}, found {h or 'absent'}")
    purged = [n for n in plan["purge"] if n in have and have[n][1].startswith(user_site)]
    for n in purged:
        print(f"[r50_overlay]    LEFTOVER {n} is still in the overlay but must come from the module")
    shadowed = [(n, have[n][1]) for n in sorted(ours) if n in have and not have[n][1].startswith(user_site)]
    for n, loc in shadowed:
        print(f"[r50_overlay]    SHADOWED {n}: the version in use lives in {loc}, not in the overlay")
    bad_import = []
    for n, mod in KEY_MODULES.items():
        if n not in ours:
            continue
        try:
            m = __import__(mod)
            f = str(getattr(m, "__file__", ""))
            if not f.startswith(user_site):
                bad_import.append((mod, f))
        except Exception as e:                                 # noqa: BLE001
            bad_import.append((mod, f"import failed: {type(e).__name__}: {e}"))
    for mod, f in bad_import:
        print(f"[r50_overlay]    IMPORT {mod}: {f}")
    # torch <-> numpy must work: with a NumPy the module's torch was not compiled against, torch only WARNS at import and from_numpy / .numpy() fail later
    probe = ("import warnings, sys\n"
             "with warnings.catch_warnings(record=True) as w:\n"
             "    warnings.simplefilter('always')\n"
             "    import torch, numpy as np\n"
             "bad = [str(x.message).strip().splitlines()[0] for x in w if 'numpy' in str(x.message).lower()]\n"
             "a = torch.from_numpy(np.arange(4.0)); b = torch.ones(3).numpy()\n"
             "print('numpy', np.__version__, 'torch', torch.__version__, '<->', 'BROKEN: ' + bad[0] if bad else 'ok')\n"
             "sys.exit(1 if bad else 0)\n")
    br = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    bridge_bad = br.returncode != 0
    print(f"[r50_overlay]    torch<->numpy bridge: {(br.stdout.strip().splitlines() or [br.stderr.strip().splitlines()[-1] if br.stderr.strip() else '?'])[-1]}")
    rt_bad = []
    for mod in ("numpy", "scipy", "pandas", "cv2", "PIL", "torchvision"):
        try:
            __import__(mod)
        except Exception as e:                                 # noqa: BLE001
            rt_bad.append(mod)
            print(f"[r50_overlay]    IMPORT {mod}: import failed: {type(e).__name__}: {e}")
    # modules transformers imports lazily (an AutoProcessor / CLIP / Qwen load goes through them) only when an optional package is installed
    import importlib.util
    for mod, dep in (("transformers.tokenization_mistral_common", "mistral_common"), ("diffusers.loaders.peft", "peft")):
        if importlib.util.find_spec(dep) is None:
            continue
        try:
            __import__(mod)
        except Exception as e:                                 # noqa: BLE001
            rt_bad.append(mod)
            print(f"[r50_overlay]    IMPORT {mod}: import failed: {type(e).__name__}: {e}")
    chk = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True).stdout.splitlines()
    Path(plan_path).with_name("overlay_pipcheck.txt").write_text("\n".join(chk) + "\n")
    # only complaints MADE BY a package we installed count (its own requirements are unmet); the module's packages (ultralytics, gradio,
    # google-api...) complaining about our newer pins are not used by R49 and are only counted
    mine = [l for l in chk if l.split() and norm(l.split()[0]) in ours]
    for l in mine[:8]:
        print(f"[r50_overlay]    PIP-CHECK {l}")
    import torch
    print(f"[r50_overlay]    runtime: python {sys.version.split()[0]}, torch {torch.__version__} (cuda {torch.version.cuda}), "
          f"{len(ours)} packages from the overlay, pip check: {len(chk)} complaint(s), {len(mine)} made by packages we installed "
          f"({len(chk) - len(mine)} by the module's own packages: informational, listed in overlay_pipcheck.txt)")
    problems = len(mism) + len(shadowed) + len(bad_import) + len(mine) + len(purged) + len(rt_bad) + int(bridge_bad)
    print(f"[r50_overlay]    MISMATCHES={len(mism)} PROBLEMS={problems}")
    return problems


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "plan":
        cmd_plan(*sys.argv[2:6])
    elif c == "wheel":
        cmd_wheel(sys.argv[2])
    elif c == "verify":
        sys.exit(min(cmd_verify(*sys.argv[2:4]), 100))
PYEOF

# ---- 2. plan
say "2. plan: ${FREEZE}"
PLAN=$ENVDIR/overlay_plan.json
REQ=$TMPD/requirements.txt
python "$TOOLS" plan "$FREEZE" "$REQ" "$PLAN" "$USER_SITE" || { say "FATAL: could not build the plan"; exit 1; }
N_INSTALL=$(grep -c . "$REQ")
WHEEL_VER=$(python -c "import json; w = json.load(open('$PLAN'))['wheel']; print(w['version'] if w else '')")
# flash-attn: say NOW (also in --dry-run) whether the wheel this module needs exists and is reachable, instead of failing midway
if [ -n "$WHEEL_VER" ]; then
  mapfile -t WH < <(python "$TOOLS" wheel "$WHEEL_VER")
  if [ -s "$WHEELS/${WH[0]}" ]; then
    say "   flash-attn wheel ${WH[0]}: already in ${WHEELS}"
  else
    HTTP=$(curl -sIL -m 25 -o /dev/null -w '%{http_code}' "${WH[1]}" 2>/dev/null); HTTP=${HTTP:-000}
    case "$HTTP" in
      200) say "   flash-attn wheel ${WH[0]}: reachable on GitHub (HTTP 200), the install will download it" ;;
      404) say "   flash-attn wheel ${WH[0]}: NO SUCH WHEEL (HTTP 404): none is published for this module's torch / CUDA / python / C++ ABI -> choose another JZ_MODULE, or build one elsewhere and put it in ${WHEELS}/"
           [ "$DRY" -eq 1 ] && FAILED=$((FAILED + 1)) ;;
      000) say "   flash-attn wheel ${WH[0]}: GitHub NOT reachable from this node -> download ${WH[1]} on a machine with access, copy it to ${WHEELS}/, then rerun"
           [ "$DRY" -eq 1 ] && FAILED=$((FAILED + 1)) ;;
      *)   say "   flash-attn wheel ${WH[0]}: unexpected HTTP ${HTTP} for ${WH[1]}"
           [ "$DRY" -eq 1 ] && FAILED=$((FAILED + 1)) ;;
    esac
  fi
fi
if [ "$DRY" -eq 1 ]; then say "dry run: nothing installed. failures: ${FAILED}"; exit $(( FAILED > 0 )); fi
PIP=(python -m pip install --user --no-deps --ignore-installed --no-cache-dir --prefer-binary --disable-pip-version-check)

# ---- 2b. things in the overlay that must come from the module (e.g. the numpy 2.x of an earlier run)
mapfile -t PURGE < <(python -c "import json; print('\n'.join(json.load(open('$PLAN'))['purge']))")
for P in "${PURGE[@]}"; do
  [ -n "$P" ] || continue
  say "2b. removing ${P} from the overlay (it is left to the module)"
  python -m pip uninstall -y "$P" > "$TMPD/pip_rm.log" 2>&1 || { say "   FAILED to remove ${P}: $(tail -n 1 "$TMPD/pip_rm.log")"; FAILED=$((FAILED + 1)); }
done

# ---- 3. pinned packages
say "3. install ${N_INSTALL} pinned packages into the overlay"
if [ "$N_INSTALL" -gt 0 ]; then
  if ! "${PIP[@]}" -r "$REQ" > "$TMPD/pip_bulk.log" 2>&1; then
    say "   bulk install failed (last lines below); retrying one package at a time"
    tail -n 4 "$TMPD/pip_bulk.log" | sed 's/^/[r50_overlay]      /'
    while read -r P; do
      "${PIP[@]}" "$P" > "$TMPD/pip_one.log" 2>&1 || { say "   FAILED to install ${P}: $(tail -n 1 "$TMPD/pip_one.log")"; FAILED=$((FAILED + 1)); }
    done < "$REQ"
  fi
else
  say "   nothing to install"
fi

# ---- 4. flash-attn
say "4. flash-attn"
if [ -n "$WHEEL_VER" ]; then
  mapfile -t WH < <(python "$TOOLS" wheel "$WHEEL_VER")
  WNAME=${WH[0]}; WURL=${WH[1]}
  say "   wanted: ${WNAME}"
  if [ ! -s "$WHEELS/$WNAME" ]; then
    say "   downloading ${WURL}"
    if ! curl -fL --retry 2 --connect-timeout 20 -o "$TMPD/$WNAME" "$WURL" 2> "$TMPD/curl.log"; then
      say "   FAILED to download the wheel ($(tail -c 160 "$TMPD/curl.log" | tr '\n' ' '))"
      say "   -> fetch it on a machine with access, copy it to ${WHEELS}/ and rerun; or choose another JZ_MODULE (torch / CUDA / python must match a published wheel)"
      FAILED=$((FAILED + 1))
    elif [ "$(stat -c %s "$TMPD/$WNAME")" -lt 10000000 ]; then
      say "   FAILED: the downloaded file is only $(stat -c %s "$TMPD/$WNAME") bytes (not a wheel)"; FAILED=$((FAILED + 1))
    else
      mv "$TMPD/$WNAME" "$WHEELS/$WNAME"
    fi
  fi
  if [ -s "$WHEELS/$WNAME" ]; then
    "${PIP[@]}" "$WHEELS/$WNAME" > "$TMPD/pip_fa.log" 2>&1 || { say "   FAILED to install the wheel: $(tail -n 1 "$TMPD/pip_fa.log")"; FAILED=$((FAILED + 1)); }
    python -c "import flash_attn; print('[r50_overlay]    flash_attn', flash_attn.__version__, 'imports from', flash_attn.__file__)" 2>&1 | tail -n 1
    python -c "import flash_attn" 2>/dev/null || { say "   FAILED: flash_attn does not import after installation (ABI mismatch with the module's torch?)"; FAILED=$((FAILED + 1)); }
  fi
else
  say "   the module already provides our version: kept"
fi

# ---- 5. the wrapper
say "5. wrapper ${WRAP}"
cat > "$WRAP" <<EOF
#!/bin/bash
# Generated by jz/r50_pip_overlay.sh ($(date '+%F %T')): python of the IDRIS module ${MODULE} + the pinned overlay. Do not edit; rerun the script.
if ! type module >/dev/null 2>&1; then
  for F in /usr/share/Modules/init/bash /etc/profile.d/modules.sh "\${MODULESHOME:-/nonexistent}/init/bash"; do [ -f "\$F" ] && { . "\$F"; break; }; done
fi
type module >/dev/null 2>&1 || { echo "r50 python wrapper: no 'module' command" >&2; exit 97; }
module purge >/dev/null 2>&1
module load arch/h100 ${MODULE} >/dev/null 2>&1 || { echo "r50 python wrapper: 'module load arch/h100 ${MODULE}' failed" >&2; exit 98; }
export PYTHONUSERBASE=${ENVDIR}
# IDRIS sets XDG_CACHE_HOME=\$WORK/.cache, which HF (HF_HOME) and torch (TORCH_HOME) prefer over ~/.cache: the shipped caches live in \$WORK/cache
export HF_HOME=${WORK}/cache/huggingface TORCH_HOME=${WORK}/cache/torch
export PYTHONPATH=${USER_SITE}:${IQA}\${PYTHONPATH:+:\$PYTHONPATH}
exec python "\$@"
EOF
chmod +x "$WRAP"
[ -d "$IQA" ] || { say "   WARNING: ${IQA} missing (pyiqa / NIQE): run jz/r50_setup_dirs.sh"; }

# ---- 6. verify through the wrapper
say "6. verify through the wrapper"
VOUT=$("$WRAP" "$TOOLS" verify "$PLAN" "$USER_SITE" 2>&1); VRC=$?
echo "$VOUT" | grep -v '^\[r50_overlay\]    MISMATCHES=' | sed -e 's/^/ /' | sed -e 's/^ \[r50_overlay\]/[r50_overlay]/'
MISM=$(echo "$VOUT" | grep -oE 'MISMATCHES=[0-9]+' | tail -1 | cut -d= -f2); MISM=${MISM:-0}
[ "$VRC" -ne 0 ] && FAILED=$((FAILED + VRC))
[ -z "$(echo "$VOUT" | grep -E 'runtime:')" ] && { say "   the wrapper did not run the verification (see above)"; FAILED=$((FAILED + 1)); MISM=${MISM:-1}; }

NFILES=$(find "$ENVDIR" -type f | wc -l)
SIZE=$(du -sh "$ENVDIR" 2>/dev/null | cut -f1)
{
  echo "r50_pip_overlay $(date '+%F %T') on $(hostname)"; echo "module: arch/h100 ${MODULE}; python ${PYV}; user site ${USER_SITE}"
  echo "freeze: ${FREEZE}"; echo "installed from the plan:"; python -c "import json; [print('  %s==%s (module had %s)' % (i['name'], i['ours'], i['module'])) for i in json.load(open('$PLAN'))['install']]"
  echo "flash-attn wheel: ${WHEEL_VER:-kept from the module}"; echo "verification: mismatches ${MISM}, failures ${FAILED}"
  echo "files in the overlay: ${NFILES} (${SIZE})"
} > "$ENVDIR/overlay.txt"
rm -f "$REQ" "$TMPD"/pip_*.log "$TMPD"/curl.log
INODES_NOTE=""; [ "$NFILES" -gt 30000 ] && INODES_NOTE=" (more than the ~5-15k expected: check idr_quota_user)"
say "overlay: ${NFILES} files, ${SIZE}${INODES_NOTE}; report ${ENVDIR}/overlay.txt"
say "installed ${N_INSTALL}$([ -n "$WHEEL_VER" ] && echo ' + flash-attn'), mismatches ${MISM}, files ${NFILES}"
say "failures: ${FAILED}"
[ "$FAILED" -eq 0 ] && say "next: bash jz/r50_port.sh"
exit $(( FAILED > 0 ))
