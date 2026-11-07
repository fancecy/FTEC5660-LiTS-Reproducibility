"""Seed the original LiTS CLI and keep all caches inside this workspace."""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "hf")
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/cert.pem"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "mpl-cache")
sys.path[:0] = [str(ROOT), str(ROOT.parent), str(ROOT.parent / "demos")]

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--mode", choices=["smoke", "search", "eval"], default="smoke")
args, remaining = p.parse_known_args()

import numpy as np
import torch
torch.set_num_threads(4)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(args.seed)

if args.mode == "smoke":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    t0 = time.perf_counter()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype,
                                               local_files_only=True).to(device)
    model.eval()
    prompt = tok.apply_chat_template([
        {"role": "user", "content": "How many positive whole-number divisors does 196 have? Explain briefly and end with the numeric answer."}
    ], tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    if device == "mps": torch.mps.synchronize()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    if device == "mps": torch.mps.synchronize()
    seconds = time.perf_counter() - t1
    text = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    record = dict(model="Qwen/Qwen2.5-1.5B-Instruct",
                  revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                  seed=args.seed, device=device, load_seconds=t1-t0,
                  generation_seconds=seconds,
                  output_tokens=int(out.shape[1]-inputs.input_ids.shape[1]), text=text)
    (ROOT / "smoke_result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
else:
    # Only the pinned local model is supported by this study. Avoid checking a
    # moving HF main revision during search; generation must use the cached pin.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import lits.lm as lm
    from huggingface_hub import constants as hf_constants
    hf_constants.HF_HUB_OFFLINE = True
    pinned = ROOT / "hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    original_exists = lm.model_exists_on_hf
    lm.model_exists_on_hf = lambda name: (
        pinned.exists() if name == "Qwen/Qwen2.5-1.5B-Instruct" else original_exists(name))
    sys.argv = ["lits-" + args.mode] + remaining
    runpy.run_module("lits.cli.search" if args.mode == "search" else "lits.cli.eval_search",
                     run_name="__main__")
