#!/bin/bash
# CPU-only portability check of Ganesh's release files, run inside a staged clean copy of the repository.
#   bash Ganesh/release/check_portable.sh <staged repo root>      (PY=<python> to choose the interpreter)
# Refuses to run if C0v2's merged weights (Ganesh/out/c0v2-target-merged/model.safetensors) are present, a sign of
# the lab workspace; it does not scan for other weights. No GPU, no downloads (HF_HUB_OFFLINE=1).
set -u
STAGE=${1:?usage: check_portable.sh <staged repo root>}
PY=${PY:-python}
cd "$STAGE" || exit 1
[ -e Ganesh/out/c0v2-target-merged/model.safetensors ] && { echo "refusing: Ganesh/out/c0v2-target-merged/model.safetensors present (lab workspace?), not a clean staged copy"; exit 1; }
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

# 1. Every Python CLI parses its arguments (imports resolve from the staged layout).
for s in train_c0 unlearn_c2 verify_merge r2_rescore compare_regen test_card_snippet; do
  if [ "$s" = test_card_snippet ]; then
    "$PY" -c "import ast,sys; ast.parse(open('Ganesh/scripts/$s.py').read())" && ok "$s.py parses" || bad "$s.py"
  elif "$PY" "Ganesh/scripts/$s.py" --help > /dev/null 2>&1; then ok "$s.py --help"; else bad "$s.py --help"; fi
done

# 2. Configs: pinned revisions, and the local inputs they name.
"$PY" - <<'EOF' || fail=1
import re, yaml, os
c0, c2 = (yaml.safe_load(open(f"Ganesh/configs/{n}.yaml")) for n in ("c0_v2", "c2"))
pins = [c0["model"]["revision"], c0["data"]["revision"], c2["model"]["tokenizer_revision"], c2["data"]["retain_revision"]]
assert all(re.fullmatch(r"[0-9a-f]{40}", p) for p in pins), pins
print("ok   configs pin 40-hex revisions for the base model and TOFU")
assert os.path.exists(c2["data"]["forget"]), c2["data"]["forget"]
print("ok   c2 forget file present:", c2["data"]["forget"])
t = c2["model"]["target"]
print(("ok   " if os.path.exists(t) else "note ") + f"c2 target {t}: " +
      ("present" if os.path.exists(t) else "absent; run train_c0.py --config Ganesh/configs/c0_v2.yaml first, or download C0v2 there"))
EOF

# 3. The merge checker's own CPU test.
"$PY" Ganesh/scripts/test_verify_merge.py > /dev/null 2>&1 && ok "test_verify_merge.py" || bad "test_verify_merge.py"

# 4. No absolute local paths in Ganesh's code, configs or docs.
hits=$(grep -rln --exclude=check_portable.sh "/home/" Ganesh/scripts Ganesh/configs Ganesh/release Ganesh/writeup 2>/dev/null)
[ -z "$hits" ] && ok "no /home/ paths in Ganesh code/configs/docs" || bad "absolute paths in: $hits"

# 5. Reads of lab-only outputs (documented as historical, not part of any route in the cards).
echo "info lab-output reads: $(grep -ln 'Vedika/out\|tofu_c2grid' Ganesh/scripts/*.py | tr '\n' ' ')"
[ $fail = 0 ] && echo "all checks passed" || echo "some checks failed"
exit $fail
