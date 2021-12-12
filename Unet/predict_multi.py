import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()

    original_img = BasicDataset.preprocess(full_img, scale_factor)
    #custom_img = BasicDataset.preprocess(custom_img, scale_factor)
    if (False):
        img = torch.from_numpy(np.concatenate((original_img,custom_img), axis=0))
    else:
        img = torch.from_numpy(original_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)
        #probs = probs.squeeze(0)

        full_mask = probs.cpu().numpy()
        full_mask = np.argmax(full_mask, 1).squeeze(0)
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)
    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask*20).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=21)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    #for i, fn in enumerate(os.listdir(folder)):
    filename = in_files
    logging.info("\nPredicting image {} ...".format(filename))
    print(filename)
    #fn = os.path.join(folder,fn)
    fn = filename[0]
        #custom = Image.open("./testCustom/" + filename +"_test.png")

    img = Image.open(fn)

    mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)
    print(mask.shape)

    plt.imshow(mask)
    plt.show()
    if not args.no_save:
        out_fn = "test.png"
        result = mask_to_image(mask)
        result.save(out_fn)
        logging.info("Mask saved to {}".format(out_fn))

    if args.viz:
        logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        plot_img_and_mask(img, mask)