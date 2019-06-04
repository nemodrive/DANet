import cv2
import glob
import argparse
import os
import random
import numpy as np
from shutil import copyfile
from cityscapesscripts.helpers.labels import name2label


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize nemodrive dataset.',
        epilog='Example: python prepare_nemodrive.py.py dataset_folder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', type=str, default=None, help='dataset directory on disk')
    parser.add_argument('out_dir', type=str, default=None, help='generate out_folder')
    parser.add_argument('--split', type=float, default=0.8, help='Training split factor')
    parser.add_argument('--ext', type=str, default="-frame.png", help='img extension')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    img_paths = glob.glob(f"{args.dir}/**/*{args.ext}", recursive=True)
    no_imgs = len(img_paths)
    out_dir = f"{args.out}/cityscapes"

    print(f"Found: {len(img_paths)}")

    out_img_folder = "out_images"
    out_dir_imgs = f"{out_dir}/{out_img_folder}"
    assert not os.path.isdir(out_dir), "Export dir exists"
    assert not os.path.isdir(out_dir_imgs), "Export dir out_dir_imgs exists"
    os.mkdir(out_dir)
    os.mkdir(out_dir_imgs)

    # Shuffle data and split
    no_train = int(len(img_paths) * args.split)
    random.shuffle(img_paths)
    train_paths = img_paths[:no_train]
    test_paths = img_paths[no_train:]

    train_file = open(f"{out_dir}/train_fine.txt", "w")
    test_file = open(f"{out_dir}/val_fine.txt", "w")

    camera_img_suff, file_type_extension = args.ext.split(".")

    road_color = np.array(name2label["road"].color, dtype=np.uint8)
    road_id = name2label["road"].trainId
    idx = 0
    for img_paths, log_file in [(train_paths, train_file), (test_paths, test_file)]:
        for x in img_paths:
            dir_source = os.path.dirname(x)
            dir_name = os.path.basename(dir_source)
            file_name = os.path.basename(x)

            label_img_name = file_name.replace(camera_img_suff, "")
            label_img_path = f'{dir_source}/{file_name.replace(camera_img_suff, "")}'

            # Label image transform to class color
            img = cv2.imread(label_img_path)
            select_class = np.all(img[:, :] == np.array([255, 255, 255], dtype=np.uint8), axis=2)
            img[select_class] = road_color

            # Generate train img
            out_train_img = np.zeros(img.shape[:2], dtype=np.uint8)
            out_train_img.fill(255)
            out_train_img[select_class] = road_id

            # Rename images (no groups)
            label_out_path = f"{out_img_folder}/{dir_name}_{label_img_name}"
            train_out_path = f"{out_img_folder}/{dir_name}_trainIds_{label_img_name}"
            photo_out_path = f"{out_img_folder}/{dir_name}_{file_name}"

            # Move images to out folder
            cv2.imwrite(f"{out_dir}/{label_out_path}", img)
            cv2.imwrite(f"{out_dir}/{train_out_path}", out_train_img)
            copyfile(x, f"{out_dir}/{photo_out_path}")

            # Write to label txt files
            log_file.write(f"{photo_out_path}\t{train_out_path}\n")
            idx += 1

            if idx % 100 == 0:
                print(f"Done: {idx}/{no_imgs}")
    train_file.close()
    test_file.close()
