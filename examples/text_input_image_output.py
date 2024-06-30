import torch
import uuid
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
    instruction1 = "Draw a flower"
    instruction2 = "Draw a horse"
    batch_prompt_ui=[
        [{"type": "text", "value": instruction1}, {"type": "sentinel", "value": "<END-OF-TURN>"}],
        [{"type": "text", "value": instruction2}, {"type": "sentinel", "value": "<END-OF-TURN>"}]
    ]
    tokens = model.generate(batch_prompt_ui=batch_prompt_ui)

    import ipdb; ipdb.set_trace()
    image_tokens = []
    for token in tokens.tolist()[0]:
        if token >= 4 and token <= 8195:
            image_tokens.append(token)
    image_tokens = image_tokens[:1024]
    print(len(image_tokens))
    image_tokens = torch.tensor(image_tokens, device="cuda")[None]

    image = model.decode_image(image_tokens)[0] # type: ignore
    image_path = f"generated_images/{instruction1}-{uuid.uuid4()}.png"
    image.save(image_path)
    print(model.decode_text(tokens)[0])
    print(f"===========================" * 2)
    print(f"Save generated images to {image_path}")


if __name__ == "__main__":
    main()
