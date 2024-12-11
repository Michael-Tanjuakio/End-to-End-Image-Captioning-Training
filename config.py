from dataclasses import dataclass
import torch

@dataclass
class parameters:

    # Hyperparameters
    max_iters = 25 # 10
    eval_interval = 50 # 50
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    learning_rate = 8e-5
    
    '''
    Test 1-6: 1hr 28min
    Learning_rate = 5e-5
    max_iters = 10
    eval_interval = 50
    batch_size 32

    Test 7: ~25min
    Learning_rate = 8e-5
    max_iters = 10
    eval_interval = 50
    batch_size 32
    # Same as tests 3 and 6

    Test 8 
    Learning_rate = 8e-5
    max_iters = 25
    eval_interval = 50
    batch_size 32


    '''