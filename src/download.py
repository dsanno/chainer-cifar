import tarfile
import os
from six.moves.urllib import request

url_dir = 'https://www.cs.toronto.edu/~kriz/'
file_name = 'cifar-10-python.tar.gz'
save_dir = 'dataset'
tar_path = os.path.join(save_dir, file_name)

if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(tar_path):
        print('{:s} already downloaded.'.format(file_name))
    else:
        print('Downloading {:s}...'.format(file_name))
        request.urlretrieve('{:s}{:s}'.format(url_dir, file_name), tar_path)

    print('Extracting files...')
    with tarfile.open(tar_path, 'r:gz') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, save_dir)
