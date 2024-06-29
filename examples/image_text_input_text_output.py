from chameleon.inference.chameleon import ChameleonInferenceModel
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
    tokens = model.generate(
        prompt_ui=[
            {"type": "image", "value": "file:data/adult_llama.png"},
            {"type": "text", "value": "What do you see?"},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ]
    )
    print(model.decode_text(tokens)[0])


if __name__ == "__main__":
    main()
