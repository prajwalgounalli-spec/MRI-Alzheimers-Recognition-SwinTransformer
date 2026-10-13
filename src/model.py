import torch
from torch import nn
from torchvision.models import swin_t, Swin_T_Weights

NUM_CLASSES = 4

def create_model(pretrained=True):
    """
    Create a Swin Transformer-Tiny model for
    four-class Alzheimer's MRI image classification.
    """
    weights = Swin_T_Weights.DEFAULT if pretrained else None

    model = swin_t(weights=weights)

    input_features = model.head.in_features
    model.head = nn.Linear(input_features, NUM_CLASSES)

    return model


def load_checkpoint(checkpoint_path, device="cpu"):
    """
    Load a previously trained Swin Transformer checkpoint.
    """
    checkpoint_path = str(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    model = create_model(pretrained=False)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Swin Transformer-Tiny created successfully.")
    print("Number of classes:", NUM_CLASSES)
    print("Output layer:", model.head)
    print(f"Total parameters: {parameter_count:,}")

    test_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(test_input)

    print("Test input shape:", tuple(test_input.shape))
    print("Model output shape:", tuple(output.shape))

    assert output.shape == (1, NUM_CLASSES)

    print("Model verification passed.")
