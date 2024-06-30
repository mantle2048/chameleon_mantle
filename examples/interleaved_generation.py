import torch
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
    tokens: torch.LongTensor = tokens[0]  # remove batch dimension
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
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments

def main():

    model = ChameleonInferenceModel(
        model_7b_path.as_posix(),
        tokenizer_text_path.as_posix(),
        tokenizer_image_cfg_path.as_posix(),
        tokenizer_image_path.as_posix(),
    )

    options = Options(max_seq_len=8192, max_gen_len=8192)
    print(options)

    instructions = [
        # "Please introduce the city of Gyumri with pictures."
        # "Please write a popular science article about 'Unraveling the Mysteries of Black Holes: A Scientific Overview' with pictures and illustrations."
        # "Please introduce China's intangible cultural heritage: Iron fireworks."
        # "Please introduce The Great Wall with some pictures."
        # ("Please tell me what you see in this image, and draw a similar picture in a comic style.", "/nas/shared/GAIR/jdsu/projects/chameleon_mantle/interleaved/bear/0.png")
        # "What color is a polar bear’s fur? Show me a photograph of the polar bear in the wild."
        # "Please tell me how to make scrambled eggs inside the shell step by step. Explain each step with text and pictures."
        # "Use pictures and words to explain each step to make a traditional egg fried rice."
        "Use pictures and words to explain each step to cook eggs."
    ]
    batch_prompt_ui = []
    for instruction in instructions:
        print(f"instruction: {instruction}, type: {type(instruction)}")
        if type(instruction) == tuple:
            inst, image_path = instruction
            batch_prompt_ui += [
                [
                    {"type": "image", "value": f"file:{image_path}"},
                    {"type": "text", "value": inst},
                    # {"type": "sentinel", "value": "<END-OF-TURN>"},
                ],
            ]
            print(batch_prompt_ui)
        else:
            batch_prompt_ui += [
                [
                    {"type": "text", "value": instruction},
                    # {"type": "sentinel", "value": "<END-OF-TURN>"},
                ],
            ]
    tokens: torch.LongTensor = model.generate(
        batch_prompt_ui=batch_prompt_ui,
        options=options
    )
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
            image_path = f"interleaved/menu/{seg_id}.png"
            img.save(image_path)
            print(f"seg {seg_id}, image: {image_path}")
        else:
            assert seg_type == "text_seg"
            decoded_text = model.decode_text(seg_tokens)[0]
            print(f"seg {seg_id}, text: {decoded_text}")

if __name__ == "__main__":
    main()
