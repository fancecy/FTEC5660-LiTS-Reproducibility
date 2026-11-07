"""Download the pinned public model once; create the expected offline cache."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "hf")
if Path("/etc/ssl/cert.pem").exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/cert.pem")
from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
snapshot = Path(snapshot_download(MODEL, revision=REVISION,
                                 cache_dir=ROOT / "hf/hub"))
ref = ROOT / "hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs/main"
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text(REVISION)
models = ROOT / "models"
models.mkdir(exist_ok=True)
link = models / "Qwen2.5-1.5B-Instruct"
if not link.exists(): link.symlink_to(snapshot, target_is_directory=True)
print(f"Pinned model ready at {snapshot}")
