import tarfile
import os
from six.moves.urllib import request

url_dir = 'https://www.cs.toronto.edu/~kriz/'
file_name = 'cifar-10-python.tar.gz'
save_dir = 'dataset'
tar_path = os.path.join(save_dir, file_name)

if __name__ == '__main__':
    if os.path.exists(tar_path):
        print('{:s} already downloaded.'.format(file_name))
    else:
        print('Downloading {:s}...'.format(file_name))
        request.urlretrieve('{:s}{:s}'.format(url_dir, file_name), tar_path)

    print('Extracting files...')
    with tarfile.open(tar_path, 'r:gz') as f:
        f.extractall(save_dir)
