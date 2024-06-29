import torch
import json
from PIL import Image
from chameleon.inference.chameleon import ChameleonInferenceModel
from chameleon.inference.constants import (
    model_7b_path,
    tokenizer_image_cfg_path,
    tokenizer_image_path,
    tokenizer_text_path,
)
import numpy as np

def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)  # [0:255] => [-1:1]
    image = torch.tensor(image.transpose(2, 0, 1)[None]).to(dtype=torch.float32)
    return image


def main():
    model = ChameleonInferenceModel(
        model_7b_path.as_posix(),
        tokenizer_text_path.as_posix(),
        tokenizer_image_cfg_path.as_posix(),
        tokenizer_image_path.as_posix(),
    )
    import ipdb; ipdb.set_trace()
    tokens = model.generate(
        prompt_ui=[
            {"type": "text", "value": "Draw a red flower."},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
            # {"type": "text", "value": "![A red flower with a touch of green around it. The flower has five pointed petals and the flower head looks fresh and pink. The flower is a delicacy that needs water and sunlight to grow. The colors are red, green and white. ](<racm3:break>"},
            # {"type": "sentinel", "value": "<END-OF-TURN>"},
        ]
    )

    # custom_image = load_image("data/red_flower.png")
    # custom_tokens =  model.token_manager.image_tokenizer.img_tokens_from_pil(custom_image)

    print(tokens)
    with open(tokenizer_text_path, "r") as fp:
        data = json.load(fp)


    token_dict = data["model"]["vocab"]

    token_list = list(token_dict.keys())
    string = ""
    image_tokens = []
    for token in tokens.tolist()[0]:
        string += token_list[token]
        if token >= 4 and token <= 8195:
            image_tokens.append(token)
    print(len(image_tokens)) # 1024
    image_tokens = torch.tensor(image_tokens, device="cuda")[None]

    image = model.decode_image(image_tokens)[0] # type: ignore
    image.save("test.png")
    print(model.decode_text(tokens)[0])


if __name__ == "__main__":
    main()
