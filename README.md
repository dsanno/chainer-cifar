# Implementation of CIFAR-10 dataset classification using Chainer

# Requirements

* Python 2.7
* [Chainer 2.0.0](http://chainer.org/)
* [Cupy 1.0.0](http://docs.cupy.chainer.org/en/stable/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [matplotlib](http://matplotlib.org/)

# How to use

## Download dataset

```
$ python src/download.py
```

## Convert dataset

```
$ python src/dataset.py
```

This command makes some dataset files:

* Image dataset files
    * dataset/image.pkl: Only mean value is subtracted
    * dataset/image_zca.pkl: Applied ZCA whitening
    * dataset/image_norm_aca.pkl: Applied Contrast normalization and ZCA whitening
* Label file
    * dataset/label.pkl

## Train

Example:
```
$ python src/train.py -g 0 -m vgg -b 128 -p vgg --optimizer adam --iter 300 --lr_decay_iter 100
```

While training the following files are saved.
* model parameter file `<prefix\>.model`
* loss and error log file `<prefix\>_log.csv`
* loss curve image `<prefix>_loss.png`
* error curve image `<prefix>_error.png`

Options:

* `-g (--gpu) <int>`: Optional  
GPU device ID. Negative value indicates CPU (default: -1)
* `-m (--model) <model name>`: Optional  
Model name that is one of the following (default: vgg)
    * `cnn`: Simple CNN
    * `cnnbn`: Simple CNN with Batch Normalization
    * `vgg`: VGG like model
    * `residual`: Residual Network
* `-b (--batch_size) <int>`: Optional  
Mini batch size (default: 100)
* `-d (--dataset) <file path>`: Optional  
File path of image dataset file (default: dataset/image.pkl)
* `-l (--label) <file path>`: Optional  
File path of label file (default: dataset/label.pkl)
* `-p (--prefix) <str>`: Optional  
Prefix of saved model file. (default: `<model name>_<optimizer name>`)
* `--iter <int>`: Optional  
Training iteration (default: 300)
* `--save_iter <int>`: Optional  
Iteration interval to save model parameter file. 0 indicates model paramete is note saved at fixed intervals. Note that the best accuracy model is always saved even if this parameter is 0. (default: 0)
* `--optimizer <str>`: Optional  
Optimizer name (`sgd` or `adam`, default: sgd)
* `--lr <float>`: Optional  
Initial learning rate for SGD (default: 0.01)
* `--alpha <float>`: Optional  
Initial alpha for Adam (default: 0.001)
* `--lr_decay_iter <int>`: Optional  
Iteration interval to decay learning rate. Learning rate is decay to 1/10 at this intervals. (default: 100)
* `--weight_decay <float>`: Optional  
Weight decay (default: 0.0001)
* `--res_depth <int>`: Optional  
Depth of Residual Network. Total number of residual blodks is `res_depth * 3` and total number of layers is `res_depth * 6 + 2` (default: 18)
* `--skip_depth`: Optional  
Use stochastic depth in Residual Network if set.
* `--seed <int>`: Optional  
Random seed (default: 1)

# License

MIT License
