import os
import glob


def change_suffix(dir, old_suffix, new_suffix):
    files = glob.glob(dir)
    for file in files:
        suffix = file.split('.')[-1]
        if suffix == old_suffix:
            new_file = file.split('.')[0] + '.' + new_suffix
            os.rename(file, new_file)
            print('{} --> {}'.format(file, new_file))


if __name__ == '__main__':
    dir = os.path.join(os.getcwd(), 'qcbj_7X_7Y\\qcbj7F2')
    # print(dir)
    change_suffix(dir + '\\*', old_suffix='jpg', new_suffix='png')
