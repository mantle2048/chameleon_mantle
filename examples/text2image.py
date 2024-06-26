import torch
import uuid
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from chameleon.inference.constants import (
    model_7b_path,
    tokenizer_image_cfg_path,
    tokenizer_image_path,
    tokenizer_text_path,
)


def main():

    model = ChameleonInferenceModel(
        model_7b_path.as_posix(),
        tokenizer_text_path.as_posix(),
        tokenizer_image_cfg_path.as_posix(),
        tokenizer_image_path.as_posix(),
    )

    options = Options()
    options.txt = False

    instructions = [
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
        "A polar bear standing on a drifting ice floe in the Arctic Ocean, with the Northern Lights in the sky.",
    ]
    batch_prompt_ui = []
    for instruction in instructions:
        batch_prompt_ui += [
            [
                {"type": "text", "value": instruction},
                {"type": "sentinel", "value": "<END-OF-TURN>"}
            ],
        ]

    image_tokens = model.generate(
        batch_prompt_ui=batch_prompt_ui,
        options=options
    )

    images =  model.decode_image(image_tokens)

    for instruction, image in zip(instructions, images):
        subdir = instruction.split(' ')[0]
        image_path = f"text2image/{subdir}-{uuid.uuid4()}.png"
        image.save(image_path)
        print(f"===========================" * 2)
        print(f"Save generated images to {image_path}")


if __name__ == "__main__":
    main()
