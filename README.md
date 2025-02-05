# slogan-transformer
Building a character-based transformer 

The dataset contains the slogans scraped from the website www.sloganlist.com  
The use of the dataset is provided for educational research only  

Using Pytorch nn.TransformerDecoder for training from scratch text generation transformer model 

### Goal
Creating a model that generates slogans using as few parameters as possible showing Transformer  
architecture can be also implemented at small scale  

### How to run the model
Running file train.py results in a training loop which fits the model. It then saves the model weights  
in .pth file (I couldn't include it in the github repository as it was too large).  

File generate.py has the option to download the model state_dict from my google drive for inference.
By running the file it will generate slogans based on starting token (can be <bos>, ' ' or any string).  

For testing I've created .ipynb file which contains every other .py file and has the same functionality  
as the whole program.

### Dependencies
The versions of the modules used in the program are in requirements.txt file  

### License
Although the usage of the code provided is open-source the dataset is bounded by non-commercial use licence  
