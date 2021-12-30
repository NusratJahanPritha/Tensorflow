
# Assignment Title

PyTorch: Transfer Learning and Image Classification




## Description

We’ll implement several Python scripts, including:

1.A configuration script to store important variables (config.py)

2.A dataset loader helper function (create_dataloaders.py)

3.A script to build and organize our dataset on disk such that PyTorch’s ImageFolder and DataLoader classes can easily be utilized (build_dataset.py)

4.A driver script that performs basic transfer learning via feature extraction(train_feature_extraction.py)

5.A second driver script that performs fine-tuning by replacing the fully connected (FC) layer head of a pre-trained network with a brand new, freshly initialized, FC head (fine_tune.py)

6.A final script that allows us to perform inference with our trained models (inference.py)
## Documentation

[Documentation](https://www.pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/
)


## Dataset

http://download.tensorflow.org/example_images/flower_photos.tgz
## Running Steps

1. build_dataset.py
2. train_feature_extraction.py
3. fine_tune.py
4. inference.py
## Running Tests

To run tests, run the following command 


python inference.py --model output/finetune_model.pth

Or,

python inference.py --model output/warmup_model.pth



