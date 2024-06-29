import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from chameleon.inference.constants import tokenizer_image_cfg_path, tokenizer_image_path
from chameleon.inference.vqgan import VQModel

config = OmegaConf.load(tokenizer_image_cfg_path)

ddconfig = config.model.params.ddconfig

embed_dim = config.model.params.embed_dim
n_embed = config.model.params.n_embed

ckpt_path = tokenizer_image_path

model = VQModel(
    ddconfig=ddconfig, embed_dim=embed_dim, n_embed=n_embed, ckpt_path=ckpt_path
)

print(model)


# load an image, and normalize it
def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)  # [0:255] => [-1:1]
    image = torch.tensor(image.transpose(2, 0, 1)[None]).to(dtype=torch.float32)
    return image


# save an image
def save_image(image, image_path):
    image = image.detach().cpu().numpy().transpose(0, 2, 3, 1)  # image
    image = (image + 1.0) * 127.5  # [-1:1] => [0:255]
    image = Image.fromarray(image[0].astype(np.uint8))
    image.save(image_path)


image_path = "data/red_flower.png"
image = load_image(image_path)
print(f"image.shape: {image.shape}")
(quant, _, (_, _, indices)) = model.encode(image)
print(f"quant.shape: {quant.shape}")
print(f"indices.shape: {indices.shape}")
rec_image = model.decode(quant)
print(f"reconstruct image.shape: {rec_image.shape} ")
save_path = "flintstone_rec.png"
save_image(rec_image, save_path)
print(f"save: {save_path}")
