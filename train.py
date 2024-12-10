import torch
import torch.nn as nn
import numpy as np

from config import parameters
from data_load import getDataSet
from model_load import getResNet, getGPT2LMHeadModel
from model import ImageCaptioningModel
from transformers import AdamW

def train(resnet_model, gpt2lmhead_model, pretrained, dataset):

    ''' Load in parameters/hyperparameters'''
    args = parameters
    batch_size = args.batch_size
    num_epochs = args.max_iters
    print_every = args.eval_interval
    device = args.device
    learning_rate = args.learning_rate

    ''' Create and split dataset '''
    train_dataloader, val_dataloader = getDataSet(batch_size, dataset)

    ''' Load in ResNet and GPT2 model '''
    resnet = getResNet(version=resnet_model, pretrained=pretrained).to(device)
    resnet = nn.Sequential(*list(resnet.children())[:-2]) # Remove avg pool and fc layer
    gpt2 = getGPT2LMHeadModel(version=gpt2lmhead_model, pretrained=pretrained).to(device)

    ''' Create Model with optimizer '''
    model = ImageCaptioningModel(resnet, gpt2).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    ''' Create loss list and BertScore list '''
    training_losses = np.zeros((num_epochs), dtype=np.float32)
    validation_losses = np.zeros((num_epochs), dtype=np.float32)

    ''' Train the model '''
    for epoch in range(num_epochs):

        # Train model
        model.train()
        total_loss = 0
        iteration_loss = 0

        for idx, batch in enumerate(train_dataloader, 1):
            batch = tuple(t.to(device) for t in batch)
            images, input_ids, masks = batch

            optimizer.zero_grad()

            outputs = model(images, input_ids, attention_mask = masks)
            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            iteration_loss += loss.item()

            if idx % print_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} | Iteration {idx}/{len(train_dataloader)} | "
                    f"Training Loss: {iteration_loss / print_every:.3f}")
                iteration_loss = 0

        # Esimate_loss    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                images, input_ids, masks = batch
                outputs = model(images, input_ids, attention_mask=masks)
                loss = outputs.loss
                val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {total_loss / len(train_dataloader):.3f} | "
            f"Validation Loss: {val_loss:.3f}\n")
        
        # Save both the training loss mean and validation loss mean
        training_losses[epoch] = total_loss / len(train_dataloader)
        validation_losses[epoch] = val_loss
    
    ''' Return model and data for generating results'''
    return model, val_dataloader, training_losses, validation_losses
