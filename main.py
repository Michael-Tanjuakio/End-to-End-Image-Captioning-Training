from train import train
from test import show_results, directory_exists
from data_load import getDataSetName
import sys

def main():

    ''' 
    *** BEFORE RUNNING MAIN ***
    Run data_install.py to locally store the datasets for testing
    '''

    '''
    Valid train arguments - *** Case sensitive ***

    Resnet Versions:    resnet18, resnet34, resnet50, resnet101, resnet152
    GPT Versions:       GPT2LMHeadModel (PyTorch subclass), TFGPT2LMHeadModel (keras subclass), FlaxGPT2LMHeadModel (flax subclass)
    pretrained:         True, False
    dataset:            dataset1 (Flickr8kDataset)
    '''

    ''' Example training '''

    # Set model versions, pretrain condition, and dataset
    resnet_model = "resnet50"
    gpt2lmhead_model = "GPT2LMHeadModel"
    pretrained = True
    pretrained_str = "Pretrained" if pretrained else "Trained"
    dataset = "dataset1"
    filepath = "Test_Results"
    info = f"{resnet_model}, {gpt2lmhead_model}, {getDataSetName(dataset)}, {pretrained_str}" # Set results name

    # Test if results folder already exists else end the program
    directory_exists(f"Results/{filepath}")
    print(f"Testing: {info}") # Passes test

    # Train the model with the dataset
    model, test_dataloader, training_loss_data, validation_loss_data = train(
        resnet_model=resnet_model, 
        gpt2lmhead_model=gpt2lmhead_model, 
        pretrained=pretrained, 
        dataset=dataset
    )
    
    # Saves loss graph, 5 image captions
    show_results(
        model=model, 
        val_dataloader=test_dataloader, 
        training_loss_data=training_loss_data, 
        validation_loss_data=validation_loss_data, 
        info=info, 
        filepath=filepath
    ) 

if __name__=="__main__":
    main()