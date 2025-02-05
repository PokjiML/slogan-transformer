import gdown


# Download the model state_dict
def download_model():
    url = 'https://drive.google.com/file/d/1QehB_SbypGvJfns7dsVJsdnQL38oOg3P/view?usp=sharing'
    output = 'slogan_generator.pth'

    # Download the model state_dict
    gdown.download(url, output, quiet=False)
