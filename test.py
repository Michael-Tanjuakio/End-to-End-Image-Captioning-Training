import torch

from config import parameters
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model_load import tokenizer
import random

args = parameters
device = args.device

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

def show_results(model, val_dataloader):
    # generate image caption
    images, _, _ = next(iter(val_dataloader))
    image = images[random.randint(0, len(val_dataloader) - 1)].to(device)
    img = tensor_to_image(image)

    caption = generate_caption(image, model, tokenizer)

    plt.imshow(img)
    plt.show()
    print(f"Label : {caption}")

    return

