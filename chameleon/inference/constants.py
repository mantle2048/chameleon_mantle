import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ckpt_path = Path(os.getenv("CKPT_PATH", "./data"))

model_7b_path = ckpt_path / "models" / "7b"

model_30b_path = ckpt_path / "models" / "30b"

tokenizer_text_path = ckpt_path / "tokenizer" / "text_tokenizer.json"

tokenizer_image_path = ckpt_path / "tokenizer" / "vqgan.ckpt"

tokenizer_image_cfg_path = ckpt_path / "tokenizer" / "vqgan.yaml"
