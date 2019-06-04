### Convert Cityscape dataset to 19 label images 

git clone https://github.com/freedrone/cityscapesScripts.git
* python setup.py install (check for requirements)
* e.g. sudo apt install python-tk python-qt4

set path for dataset
* export CITYSCAPES_DATASET=\<path>/gtFine_trainvaltest

Convert dataset -> should result in PNG images with colered labels
* python cityscapesscripts/preparation/createTrainIdLabelImgs.py

### Fix some stuff

__not sure if it works__

Uptades for Pytorch-Encoding
* git clone https://github.com/zhanghang1989/PyTorch-Encoding.git
* python setup.py install

### Preprocess nemodrive images & export to cityscape format 

It will search for all images with suffix --ext and will consider labeled images named without <ext> *(but with img extension).
Exports: original_image, colored_label and grayscale_trainIds_image, fine_train.txt, fine_val.txt -> {args.out}/cityscapes

python <img_folder> <out_dir>

### Eval nemodrive dataset

Set --data-root ../datasets/nemodrive

CUDA_VISIBLE_DEVICES=0 python test.py --dataset cityscapes --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 12 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval --data-root ../datasets/nemodrive

<problems with single GPU>
