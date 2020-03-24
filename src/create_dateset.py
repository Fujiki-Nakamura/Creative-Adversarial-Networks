import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
from tqdm import tqdm


def main(args):
    data_d = Path('../data/wikiart')
    output_d = Path(args.output_d)

    impath_list = sorted(list(data_d.glob('./*/*.jpg')))
    im_list = []
    imname_list = []
    style_name_list = []
    for i, impath in enumerate(tqdm(impath_list)):
        imname_list.append(impath.name)
        style_name_list.append(impath.parent.name)
        im = io.imread(impath.as_posix())
        im = resize(im, args.imsize)
        im_list.append(im)
    ima = np.stack(im_list, axis=0)
    path = output_d/f'{args.imsize[0]}x{args.imsize[1]}.npy'
    np.save(path, ima)
    print(f'Saved at {path}')

    df = pd.DataFrame()
    df['imname'] = imname_list
    df['style'] = style_name_list
    path = output_d/f'list.csv'
    df.to_csv(path, index=False)
    print(f'Saved at {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, nargs='+', default=(128, 128))
    parser.add_argument('--output_d', type=str, default='../data/')
    args, unknown_args = parser.parse_known_args()
    main(args)
