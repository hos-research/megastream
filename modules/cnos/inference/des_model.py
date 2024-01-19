# Third Party
import torch

# CNOS
from cnos.model.dinov2 import CustomDINOv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_descriptor_model(
    model_name: str,
    descriptor_width_size: int,
    token_name: str = 'x_norm_clstoken',
    image_size: int = 224,
    chunk_size: int = 16,
) -> CustomDINOv2:
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
    )
    custom_dino = CustomDINOv2(
        model_name=model_name,
        model=model,
        token_name=token_name,
        image_size=image_size,
        chunk_size=chunk_size,
        descriptor_width_size=descriptor_width_size,
    )
    custom_dino.model = custom_dino.model.to(device)
    custom_dino.model.device = device
    return custom_dino
