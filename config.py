from dataclasses import dataclass
import torch

@dataclass
class parameters:

    # Hyperparameters
    max_iters = 10
    eval_interval = 50
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    learning_rate = 5e-5

    '''
    Resnet Versions

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152

    # Note: Case sensitive

    '''
    # resnet_model = "resnet50"

    '''
    GPT Versions 

    GPT2LMHeadModel             - PyTorch subclass
    TFGPT2LMHeadModel           - keras subclass
    FlaxGPT2LMHeadModel         - flax subclass
    
    '''
    # gpt2_model = "GPT2LMHeadModel"

    


    