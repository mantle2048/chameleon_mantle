import torch
import pdb
import uuid
from PIL import Image
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from chameleon.inference.constants import (
    model_7b_path,
    tokenizer_image_cfg_path,
    tokenizer_image_path,
    tokenizer_text_path,
)

from typing import List, Tuple

def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens: torch.LongTensor = tokens[0]  # Remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
            # current_segment.append(token)
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            # current_segment.append(token)
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)

    # save any remaining tokens as a text segment
    if current_segment:
        if in_image_seg:
            segments.append(('image_seg', torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(('text_seg', torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments

def main():

    model = ChameleonInferenceModel(
        model_7b_path.as_posix(),
        tokenizer_text_path.as_posix(),
        tokenizer_image_cfg_path.as_posix(),
        tokenizer_image_path.as_posix(),
    )

    options = Options()

    instructions = [
        "Please introduce the city of Gyumri with pictures."
    ]
    batch_prompt_ui = []
    for instruction in instructions:
        batch_prompt_ui += [
            [
                {"type": "text", "value": instruction},
                # {"type": "sentinel", "value": "<END-OF-TURN>"}
            ],
        ]

    tokens: torch.LongTensor = model.generate(
        batch_prompt_ui=batch_prompt_ui,
        options=options
    )
    # torch.save(tokens, "tokens.pth")
    # tokens: torch.LongTensor = torch.load("tokens.pth")
    print(tokens.shape)
    print(tokens)
    
    # split
    boi, eoi = model.vocab.begin_image, model.vocab.end_image   # 8197(begin), 8196(end)
    segments = split_token_sequence(tokens, boi, eoi)
    # decode
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            assert seg_tokens.shape[1] == 1024
            # seg_tokens = seg_tokens.to("cpu")
            img: Image = model.decode_image(seg_tokens)[0]
            # pdb.set_trace()
            image_path = f"interleaved/{seg_id}.png"
            img.save(image_path)
        else:
            assert seg_type == "text_seg"
            decoded_text = model.decode_text(seg_tokens)[0]
            print(f"seg {seg_id}, text: {decoded_text}")

if __name__ == "__main__":
    main()
