# This file is a part of
# 
# Xiaoming Zhao, Fangchang Ma, David Güera, Zhile Ren, Alexander G. Schwing, 
# and Alex Colburn. Generative Multiplane Images: Making a 2D GAN 3D-Aware. ECCV 2022.
# https://github.com/apple/ml-gmpi


import argparse
import glob
import os

import numpy as np
import PIL
import tqdm

try:
    import pyspng
except ImportError:
    pyspng = None

# test with tensorflow-gpu==2.8.0
import tensorflow as tf
from mtcnn import MTCNN

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


N_IMGS = 1


def get_file_ext(fname):
    return os.path.splitext(fname)[1].lower()


def main(args):

    detector = MTCNN()

    detect_res_dir = os.path.join(args.data_dir, "detections")
    os.makedirs(detect_res_dir, exist_ok=True)

    print("\ndetect_res_dir: ", detect_res_dir, "\n")

    sorted_f_list = sorted(list(glob.glob(os.path.join(args.data_dir, "*.png"))))

    print("\nsorted_f_list: ", len(sorted_f_list), sorted_f_list[:5], "\n")

    for i, f_path in tqdm.tqdm(enumerate(sorted_f_list), total=len(sorted_f_list)):

        basename = os.path.splitext(os.path.basename(f_path))[0]

        if pyspng is not None and get_file_ext(f_path) == ".png":
            with open(f_path, "rb") as fin:
                img = pyspng.load(fin.read())
        else:
            img = np.array(PIL.Image.open(f_path))

        text_path = f"{detect_res_dir}/{basename}.txt"
        result = detector.detect_faces(img)
        try:
            keypoints = result[0]["keypoints"]
            with open(text_path, "w") as f:
                for value in keypoints.values():
                    f.write(f"{value[0]}\t{value[1]}\n")
                # print(f"File successfully written: {text_path}")
        except:
            if i == 0:
                mode = "w"
            else:
                mode = "a"
            with open(os.path.join(detect_res_dir, "fail_list.txt"), mode) as fail_f:
                fail_f.write(f"{os.path.basename(f_path)}\n")
            print("\n[fail] ", os.path.basename(f_path), "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get landmarks from images.")
    parser.add_argument("--data_dir", type=str, default=None, help="folder for metfaces")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main(args)
