import torch

def get_best_device_pytorch():
    device = None
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'{torch.cuda.device_count()} GPU(s) available. Using the GPU: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac ARM64 GPU")
    else:
        device = torch.device("cpu")
        print('No GPU available, using CPU')
    return device