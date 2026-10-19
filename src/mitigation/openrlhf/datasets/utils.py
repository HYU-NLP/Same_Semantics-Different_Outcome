import torch
import torch.nn.functional as F


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def zero_pad_images(images, value=0):
    max_height = max(img.size(-2) for img in images)
    max_width = max(img.size(-1) for img in images)
    padded_images = []
    for img in images:
        pad_height = max_height - img.size(-2)
        pad_width = max_width - img.size(-1)
        padding = (0, pad_width, 0, pad_height)
        padded_images.append(F.pad(img, padding, value=value))
    return torch.stack(padded_images, dim=0)


def exist_and_not_none(d, key):
    return key in d and not d[key] is None
