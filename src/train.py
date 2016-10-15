import argparse
import cPickle as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import chainer
from chainer import optimizers
from chainer import serializers
import net

import trainer

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 dataset trainer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU device ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', type=str, default='vgg', choices=['cnn', 'cnnbn', 'vgg', 'residual', 'identity_mapping', 'vgg_no_fc', 'vgg_wide', 'vgg_crelu', 'inception', 'pyramid'],
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
    parser.add_argument('--save_iter', type=int, default=0,
                        help='Iteration interval to save model parameter file.')
    parser.add_argument('--lr_decay_iter', type=str, default='100',
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
                        help='Depth of Residual Network')
    parser.add_argument('--skip_depth', action='store_true',
                        help='Use stochastic depth in Residual Network')
    parser.add_argument('--swapout', action='store_true',
                        help='Use swapout')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    log_file_path = '{}_log.csv'.format(args.prefix)
    lr_decay_iter = map(int, args.lr_decay_iter.split(','))

    print('loading dataset...')
    with open(args.dataset, 'rb') as f:
        images = pickle.load(f)
        index = np.random.permutation(len(images['train']))
        train_index = index[:-5000]
        valid_index = index[-5000:]
        train_x = images['train'][train_index].reshape((-1, 3, 32, 32))
        valid_x = images['train'][valid_index].reshape((-1, 3, 32, 32))
        test_x = images['test'].reshape((-1, 3, 32, 32))
    with open(args.label, 'rb') as f:
        labels = pickle.load(f)
        train_y = labels['train'][train_index]
        valid_y = labels['train'][valid_index]
        test_y = labels['test']

    print('start training')
    if args.model == 'cnn':
        cifar_net = net.CNN()
    elif args.model == 'cnnbn':
        cifar_net = net.CNNBN()
    elif args.model == 'residual':
        cifar_net = net.ResidualNet(args.res_depth, swapout=args.swapout, skip=args.skip_depth)
    elif args.model == 'identity_mapping':
        cifar_net = net.IdentityMapping(args.res_depth, swapout=args.swapout, skip=args.skip_depth)
    elif args.model == 'vgg_no_fc':
        cifar_net = net.VGGNoFC()
    elif args.model == 'vgg_wide':
        cifar_net = net.VGGWide()
    elif args.model == 'vgg_crelu':
        cifar_net = net.VGGCReLU()
    elif args.model == 'inception':
        cifar_net = net.Inception()
    elif args.model == 'pyramid':
        cifar_net = net.PyramidNet(args.res_depth)
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

    state = {'best_valid_error': 100, 'best_test_error': 100, 'clock': time.clock()}
    def on_epoch_done(epoch, n, o, loss, acc, valid_loss, valid_acc, test_loss, test_acc):
        error = 100 * (1 - acc)
        valid_error = 100 * (1 - valid_acc)
        test_error = 100 * (1 - test_acc)
        print('epoch {} done'.format(epoch))
        print('train loss: {} error: {}'.format(loss, error))
        print('valid loss: {} error: {}'.format(valid_loss, valid_error))
        print('test  loss: {} error: {}'.format(test_loss, test_error))
        if valid_error < state['best_valid_error']:
            serializers.save_npz('{}.model'.format(model_prefix), n)
            serializers.save_npz('{}.state'.format(model_prefix), o)
            state['best_valid_error'] = valid_error
            state['best_test_error'] = test_error
        if args.save_iter > 0 and (epoch + 1) % args.save_iter == 0:
            serializers.save_npz('{}_{}.model'.format(model_prefix, epoch + 1), n)
            serializers.save_npz('{}_{}.state'.format(model_prefix, epoch + 1), o)
        # prevent divergence when using identity mapping model
        if args.model == 'identity_mapping' and epoch < 9:
            o.lr = 0.01 + 0.01 * (epoch + 1)
        if len(lr_decay_iter) == 1 and (epoch + 1) % lr_decay_iter[0] == 0 or epoch + 1 in lr_decay_iter:
            if hasattr(optimizer, 'alpha'):
                o.alpha *= 0.1
            else:
                o.lr *= 0.1
        clock = time.clock()
        print('elapsed time: {}'.format(clock - state['clock']))
        state['clock'] = clock
        with open(log_file_path, 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(epoch + 1, loss, error, valid_loss, valid_error, test_loss, test_error))

    with open(log_file_path, 'w') as f:
        f.write('epoch,train loss,train acc,valid loss,valid acc,test loss,test acc\n')
    cifar_trainer.fit(train_x, train_y, valid_x, valid_y, test_x, test_y, on_epoch_done)

    print('best test error: {}'.format(state['best_test_error']))

    train_loss, train_acc, test_loss, test_acc = np.loadtxt(log_file_path, delimiter=',', skiprows=1, usecols=[1, 2, 5, 6], unpack=True)
    epoch = len(train_loss)
    xs = np.arange(epoch, dtype=np.int32) + 1
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_loss, label='train loss', c='blue')
    ax.plot(xs, test_loss, label='test loss', c='red')
    ax.set_xlim((1, epoch))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(loc='upper right')
    plt.savefig('{}_loss.png'.format(args.prefix), bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_acc, label='train error', c='blue')
    ax.plot(xs, test_acc, label='test error', c='red')
    ax.set_xlim([1, epoch])
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    ax.legend(loc='upper right')
    plt.savefig('{}_error'.format(args.prefix), bbox_inches='tight')
