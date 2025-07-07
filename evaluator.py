import numpy as np
import torch

from model import DigitCNN

def evaluate(input_matrix: np.ndarray) -> int:
    """
    Receives images (28x28) and spits out number prediction based on CNN saved.
    """
    MODEL_PATH = "digit_cnn.pth"

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    transform = model.transformer

    if input_matrix.shape != (28, 28):
        raise ValueError("Input matrix shape must be (28, 28)")

    img_tensor = transform(input_matrix)

    # add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    return pred

