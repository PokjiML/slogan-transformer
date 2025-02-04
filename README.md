# slogan-transformer
Building a character-based transformer 

The dataset contains the slogans scraped from the website www.sloganlist.com  
The use of the dataset is provided for educational research only  

Using Pytorch nn.TransformerDecoder for training from scratch   
text generation transformer model 

### Goal
Creating a model that generates slogans to be used by a company  
using as few parameters as possible showing Transformer architecture  
can be also implemented at small scale  

### How to run the model
By running file train.py the program should generate in the directory file 'slogan_generator.pth'  
with saved weights. (I couldn't place pth file in the github directory due too file size)  

Model can be tested in generate.py file which will download state_dict of the model and perform  
next token prediction with output.

### Dependencies
The versions of the modules used in the program are in requirements.txt file  

### License
Although the usage of the code provided is open-source the dataset is bounded by non-commercial use licence  
