"""CPU self-check for evaluate.py's resume guards. Run: python Vedika/scripts/test_evaluate.py"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402


def must_fail(fn, *a):
    try:
        fn(*a)
    except AssertionError:
        return
    raise SystemExit(f"{fn.__name__}{a!r} did not refuse")


def fake_ak(name="tiny-aya-x@/ckpt#aaa", batch_size=16, eos=ev.STOP):
    tok = type("Tok", (), {"name_or_path": ev.ORIG, "chat_template": "{{ messages }}", "__len__": lambda s: 262144})()
    kw = {"batch_size": batch_size} | ({} if eos is None else {"eos_token_id": eos})
    return SimpleNamespace(name=name, _extra_kwargs=kw, _pipeline=SimpleNamespace(tokenizer=tok))


rows = [{"id": "a", "prompt": "p1"}, {"id": "b", "prompt": "p2"}]
data = {"harmbench_heldout": rows}
tmp = Path(tempfile.mkdtemp())

# manifest: same spec resumes; another checkpoint, batch size or prompt text under the same label is refused
d = tmp / "x"
spec = ev.run_spec(fake_ak(), data)
ev.check_manifest(d, spec)
ev.check_manifest(d, spec)
must_fail(ev.check_manifest, d, ev.run_spec(fake_ak(name="tiny-aya-x@/ckpt#bbb"), data))
must_fail(ev.check_manifest, d, ev.run_spec(fake_ak(batch_size=1), data))
must_fail(ev.check_manifest, d, ev.run_spec(fake_ak(), {"harmbench_heldout": [{"id": "a", "prompt": "CHANGED"}, rows[1]]}))
ev.check_manifest(d, ev.run_spec(fake_ak(), {"xstest_safe": rows}))  # a new set joins the same manifest
assert set(ev.json.loads((d / "manifest.json").read_text())["prompts"]) == {"harmbench_heldout", "xstest_safe"}

# outputs from before manifests existed are refused, not trusted
(tmp / "legacy/cache").mkdir(parents=True)
must_fail(ev.check_manifest, tmp / "legacy", spec)

# a candidate needs an original made with the same settings: batch 1 vs the batch-16 original is refused
out = tmp / "run"
ev.check_manifest(out / "original", ev.run_spec(fake_ak(name="tiny-aya-original@hf#ccc"), data))
assert ev.check_reference(out, ev.run_spec(fake_ak(), data)) == "tiny-aya-original@hf#ccc"
must_fail(ev.check_reference, out, ev.run_spec(fake_ak(batch_size=1), data))
must_fail(ev.check_reference, tmp / "empty", spec)

# dependencies: a judge cache built from older answers is stale; ids must still match (the --limit check)
c = tmp / "deps"
c.mkdir()
ans, jud = c / "s.answers.jsonl", c / "s.judge.jsonl"
ev.write_stage(ans, [{"id": "a", "response": "OLD"}, {"id": "b", "response": "x"}])
ev.write_stage(jud, [{"id": "a", "refusal": 1}, {"id": "b", "refusal": 0}], (ans,))
assert ev.cached(jud, rows, (ans,)) is not None
ev.write_stage(ans, [{"id": "a", "response": "NEW CHECKPOINT ANSWER"}, {"id": "b", "response": "x"}])
assert ev.cached(jud, rows, (ans,)) is None, "judge built from old answers was reused"
must_fail(ev.cached, ans, rows[:1])

# checkpoint identity changes when a weight file is rewritten
w = tmp / "ckpt"
w.mkdir()
(w / "model.safetensors").write_bytes(b"1234")
before = ev.weights_id(w)
(w / "model.safetensors").write_bytes(b"123456")
os.utime(w / "model.safetensors", (1, 1))
assert ev.weights_id(w) != before

# stop IDs: STOP names the installed tokenizer's end-of-turn and EOS tokens (HF cache, offline)
from transformers import AutoTokenizer  # noqa: E402
assert AutoTokenizer.from_pretrained(ev.ORIG).convert_ids_to_tokens(ev.STOP) == ev.STOP_TOKENS

# stop IDs are part of the spec; outputs made without them (all pre-fix manifests) are neither resumed nor compared
assert spec["stop_ids"] == ev.STOP
no_stop = ev.run_spec(fake_ak(eos=None), data)
must_fail(ev.check_manifest, d, no_stop)
old = tmp / "old"
ev.check_manifest(old / "original", ev.run_spec(fake_ak(name="tiny-aya-original@hf#ccc", eos=None), data))
must_fail(ev.check_reference, old, spec)

# evaluate() refuses a model wrapped without eos_token_id=STOP before it writes anything
for bad in (None, [3]):
    must_fail(ev.evaluate, fake_ak(eos=bad), "original", tmp / "refused", ("harmbench_heldout",), 2)
assert not (tmp / "refused").exists()
print("evaluate resume and stop-ID guards: all checks passed")
