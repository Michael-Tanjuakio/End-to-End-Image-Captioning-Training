from train import train
from test import show_results

def main():

    ''' 
    *** BEFORE RUNNING MAIN ***
    Run data_install.py to locally store the datasets for testing
    '''

    '''
    Resnet Versions: resnet18, resnet34, resnet50, resnet101, resnet152
    GPT Versions: GPT2LMHeadModel (PyTorch subclass), TFGPT2LMHeadModel (keras subclass), FlaxGPT2LMHeadModel (flax subclass)
    *** Case sensitive ***
    '''

    # Example training
    model, test_dataloader = train(resnet_model="resnet50", gpt2lmhead_model="GPT2LMHeadModel", pretrained=True, dataset="dataset1")
    show_results(model, test_dataloader) # Only shows one result

if __name__=="__main__":
    main()