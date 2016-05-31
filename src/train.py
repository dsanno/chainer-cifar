import argparse
import cPickle as pickle
import numpy as np
import os
import chainer
from chainer import optimizers
from chainer import serializers
import cifar
import net
import trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 dataset trainer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str, default='vgg', choices=['cnn', 'cnn_batch', 'cnin', 'vgg', 'residual', 'identity_mapping'],
                        help='Model name')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='Mini batch size')
    parser.add_argument('--dataset', '-d', type=str, default='dataset/image.pkl',
                        help='Dataset image pkl file path')
    parser.add_argument('--label', '-l', type=str, default='dataset/label.pkl',
                        help='Dataset label pkl file path')
    parser.add_argument('--prefix', '-p', type=str, default=None,
                        help='Prefix of model parameter files')
    parser.add_argument('--iter', type=int, default=300,
                        help='Training iteration')
    parser.add_argument('--save_iter', type=int, default=20,
                        help='Iteration interval to save model parameter file')
    parser.add_argument('--lr_decay_iter', type=int, default=100,
                        help='Iteration interval to decay learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate for SGD')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Initial alpha for Adam')
    parser.add_argument('--res_depth', type=int, default=18,
                        help='Depth of Residual Net')
    parser.add_argument('--swapout', action='store_true',
                        help='Use swapout')
    parser.add_argument('--seed', type=int, default=1319,
                        help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    print('loading dataset...')
    with open(args.dataset, 'rb') as f:
        images = pickle.load(f)
        train_x = images['train'].reshape((-1, 3, 32, 32))
        test_x = images['test'].reshape((-1, 3, 32, 32))
    with open(args.label, 'rb') as f:
        labels = pickle.load(f)
        train_y = labels['train']
        test_y = labels['test']

    print('start training')
    if args.model == 'cnn':
        cifar_net = net.CNN()
    elif args.model == 'cnn_batch':
        cifar_net = net.CNNWithBatch()
    elif args.model == 'cnin':
        cifar_net = net.CNIN()
    elif args.model == 'residual':
        cifar_net = net.ResidualNet(args.res_depth, swapout=args.swapout)
    elif args.model == 'identity_mapping':
        cifar_net = net.IdentityMapping(args.res_depth, swapout=args.swapout)
    else:
        cifar_net = net.VGG()

    if args.optimizer == 'sgd':
        optimizer = optimizers.MomentumSGD(lr=args.lr)
    else:
        optimizer = optimizers.Adam(alpha=args.alpha)
    optimizer.setup(cifar_net)
    if args.weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    cifar_trainer = trainer.CifarTrainer(cifar_net, optimizer, args.iter, args.batch_size, args.gpu)
    if args.prefix is None:
        model_prefix = '{}_{}'.format(args.model, args.optimizer)
    else:
        model_prefix = args.prefix

    def on_epoch_done(epoch, n, o, loss, acc, test_loss, test_acc):
        print('epoch {} done'.format(epoch))
        print('train loss: {} acc: {}'.format(loss, acc))
        print('test  loss: {} acc: {}'.format(test_loss, test_acc))
        if (epoch + 1) % args.save_iter == 0:
            serializers.save_npz('{}_{}.model'.format(model_prefix, epoch + 1), n)
            serializers.save_npz('{}_{}.state'.format(model_prefix, epoch + 1), o)
        if (epoch + 1) % args.lr_decay_iter == 0:
            if hasattr(optimizer, 'alpha'):
                optimizer.alpha *= 0.1
            else:
                optimizer.lr *= 0.1
    cifar_trainer.fit(train_x, train_y, test_x, test_y, on_epoch_done)
