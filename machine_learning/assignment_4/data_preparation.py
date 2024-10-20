# Imports
import argparse
import os


def main()->None:
    parser = argparse.ArgumentParser(description='Data Preparation Arguments')
    parser.add_argument('--train_dir', type=str, help='Data file', default='DS_IDRID')
    args = parser.parse_args()

    sources = ['Test', 'Train']
    r_label = ['-1', '-2']
    print('Starting the data preparation process...')

    for source in sources:
        path = os.path.join(args.train_dir, source)

        # print count of the files in the directory
        print(f'Number of files in the directory: {len(os.listdir(path))}')

        for file in os.listdir(path):
            if file.endswith('-1.jpg'):
                target = os.path.join(args.train_dir, 'discarded', source, '1')
                if not os.path.exists(target):
                    os.makedirs(target)
                os.rename(os.path.join(path, file), os.path.join(target, file))

            elif file.endswith('-2.jpg'):
                target = os.path.join(args.train_dir, 'discarded', source,'2')
                # create if target directory does not exist
                if not os.path.exists(target):
                    os.makedirs(target)
                os.rename(os.path.join(path, file), os.path.join(target, file))

    #



if __name__ == "__main__":
    main()
