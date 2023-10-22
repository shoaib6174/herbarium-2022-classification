import argparse
import logging
import os
import os.path
import sys
from multiprocessing import Pool

# import tensorflow as tf
import cv2
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def resize_image(fpaths):
    # Ideally we would take 2 separate args but then we would need to
    # use Pool.starmap instead of Pool.imap and Pool.starmap does not
    # play well with tqdm.
    in_fpath, out_fpath = fpaths
    src = cv2.imread(in_fpath)
    if src is None:
        _logger.warn("Failed to cv2.imread '%s' so skipping it", in_fpath)
        return
    borderType = cv2.BORDER_REFLECT_101
    # get the h and w
    h, w = src.shape[:2]
    src = src[20 : h - 20, 20 : w - 20, :]
    if h > w:
        right = h - w
        image = cv2.copyMakeBorder(src, 0, 0, 0, right, borderType)
    elif w > h:
        top = w - h
        image = cv2.copyMakeBorder(src, top, 0, 0, 0, borderType)
    else:
        image = src
    
    # resize
    image = cv2.resize(image, (512,512))
    # save image:
    cv2.imwrite(out_fpath, image)


def main():
    path = os.getcwd()
    parent = os.path.dirname(path)

    for train_or_test in ["test","train"]:
        in_dir = f"{parent}/herbarium2022/{train_or_test}_images/"

        out_dir = in_dir

        in_filelist= f"{train_or_test}_img_files.txt"
        starmap_args = []
        required_out_dirs = set()
        with open(in_filelist, "r") as fh:
            for line in fh:
                rel_fpath = line.strip()
                in_fpath = os.path.join(in_dir, rel_fpath)
                out_fpath = os.path.join(out_dir, rel_fpath)
                required_out_dirs.add(os.path.dirname(out_fpath))
                starmap_args.append((in_fpath, out_fpath))

 

        with Pool(1) as pool:
            # We use imap instead of starmap because tqdm does not play
            # easily with starmap.
            for _ in tqdm(
                pool.imap_unordered(resize_image, starmap_args),
                total=len(starmap_args),
            ):
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())