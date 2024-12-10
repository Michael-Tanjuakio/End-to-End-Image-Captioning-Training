from textwrap import wrap
import torch
import numpy as np

from config import parameters
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model_load import tokenizer
import random
import os
from pathlib import Path
import sys

args = parameters
device = args.device

def directory_exists(filepath):
    if Path(filepath).is_dir():
        print("Results folder already exists, please rename")
        sys.exit(1)

''' Print loss graph'''

def plot_losses(training_loss_data, validation_loss_data, info, filepath):
    
    # filename="Results/training_loss.pdf"

    tld = training_loss_data
    vld = validation_loss_data
    num_epoches = tld.shape[0]
    # l = np.mean(losses, axis=1)

    plt.subplot(1, 1, 1)
    plt.plot(range(num_epoches), tld, color='blue', label=f"Training Loss: {training_loss_data[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print training loss data
    plt.plot(range(num_epoches), vld, color='orange', label=f"Validation Loss: {validation_loss_data[-1]:.3f}", marker='o', alpha=0.5, ms=4) # print validation loss data
    plt.title(f"Loss - {info}")
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(6, 4)

    loss_graph_path = f"{filepath}/Loss_Graph.pdf"
    plt.savefig(loss_graph_path, bbox_inches='tight')
    print('Saved Loss Graph to %s' % (loss_graph_path))
    plt.clf()

''' Test everything and print everything in files'''

def tensor_to_image(tensor):
    # Convert the tensor to a PIL Image
    return transforms.ToPILImage()(tensor)

def generate_caption(image, model, tokenizer, max_length=50, temperature=1.0):

    caption = [tokenizer.bos_token_id]

    model.eval()

    with torch.no_grad():
        for i in range(max_length):
            input_ids = torch.LongTensor(caption).unsqueeze(0)
            input_ids = input_ids.to(device)
            outputs = model(image.unsqueeze(0), input_ids, attention_mask=None)

            logits = outputs.logits[:, -1, :] / temperature
            predicted_id = logits.argmax(1).item()
            caption.append(predicted_id)

            if predicted_id == tokenizer.eos_token_id and i>1:
                break
    generated_caption = tokenizer.decode(caption, skip_special_tokens=True)
    return generated_caption

def show_results(model, val_dataloader, training_loss_data, validation_loss_data, info, filepath):

    ''' Save loss graph in results folder '''
    results_folder_path = f"Results/{filepath}"
    os.mkdir(results_folder_path) # Create result set in results folder
    plot_losses(training_loss_data, validation_loss_data, info, results_folder_path)

    ''' Generate 5 Image Captions per result'''
    image_captions_path = f"{results_folder_path}/ImageCaptions"
    os.mkdir(image_captions_path) # Make image directory
    for i in range(1, 6):

        # Retrieve Random Image
        images, _, _ = next(iter(val_dataloader))
        image = images[random.randint(0, len(val_dataloader) - 1)].to(device)
        img = tensor_to_image(image)

        # Generate Caption
        caption = generate_caption(image, model, tokenizer)

        # Create figure
        ax = plt.figure().add_subplot(111)

        # Add image
        ax.imshow(img)

        # Add generated caption
        plt.title(caption, loc='center', wrap=True)

        # Save figure on results set folder
        img_cap_path = f"{image_captions_path}/img_cap_{i}.pdf"
        plt.savefig(img_cap_path)
        plt.clf()