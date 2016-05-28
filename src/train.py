import cPickle as pickle
import numpy as np
import os
import chainer
from chainer import optimizers
from chainer import serializers
import cifar
import net
import trainer

def on_epoch_done(epoch, n, o, loss, acc, test_loss, test_acc):
    print('epoch {} done'.format(epoch))
    print('train loss: {} acc: {}'.format(loss, acc))
    print('test  loss: {} acc: {}'.format(test_loss, test_acc))
    if (epoch + 1) % 25 == 0:
        optimizer.lr *= 0.1

if __name__ == '__main__':
    dataset_path = 'dataset'
    device_id = 0

    print('loading dataset...')
    with open(os.path.join(dataset_path, 'image.pkl'), 'rb') as f:
        images = pickle.load(f)
        train_x = images['train'].reshape((-1, 3, 32, 32))
        test_x = images['test'].reshape((-1, 3, 32, 32))
    with open(os.path.join(dataset_path, 'label.pkl'), 'rb') as f:
        labels = pickle.load(f)
        train_y = labels['train']
        test_y = labels['test']

    print('start training')
    cifar_net = net.ConvWithBatch()
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(cifar_net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    cifar_trainer = trainer.CifarTrainer(cifar_net, optimizer, 100, 100, device_id)

    cifar_trainer.fit(train_x, train_y, test_x, test_y, on_epoch_done)
