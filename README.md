# End-to-End Image-Captioning Training

## Implementation Details
This program takes a ResNet and GPT2 model and trains on a dataset to output a predicted image with a predicted caption

We use the CIFAR10 Dataset  

ResNet models are from PyTorch libraries  
- Pretrained ResNet weights are from training a ImageNet set

GPT2 models are from Hugging Face libraries  
- Pretrained GPT2 weights comes from training very large corpus of ~40 GB of text data  

## Running the program
1. **Run data_install.py** to locally download datasets for training 
2. **Edit main.py** for desired ResNet version, GPT2 version, dataset, pretrained mode
3. **Run main.py** to test the results

### To do:
- [ ] Put results in a folder instead of directly displaying after training
- [ ] Add 4 more datasets for testing
- [ ] Implement BertScore and Results Graph

## Credits
Nicolas Deperrois 
- https://www.kaggle.com/code/ndeperrois/resnet-gpt2
- https://colab.research.google.com/drive/1SVd9oy4rmt6avoDbnS2aIddiLtf77wf7#scrollTo=KDc3szmqlyxb
- _This was a modification of the owner's code which includes scalable testing across different language models/datasets_
- _"This Notebook has been released under the Apache 2.0 open source license."_

TorchVision Models  
- https://github.com/pytorch/vision  

HuggingFace OpenAI Models
- https://huggingface.co/docs/transformers/en/model_doc/gpt2  

## Owners of Repository
Michael Tanjuakio, Sowresh, Artan, Anthony
