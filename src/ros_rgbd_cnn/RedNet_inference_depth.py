import argparse
import torch
import imageio
import skimage.transform
import torchvision

import torch.optim
import RedNet_model_depth
import RedNet_model
from utils import utils
from utils.utils import load_ckpt

import os
import skimage.io
import glob



parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480


def inference():

    model = RedNet_model_depth.RedNet(pretrained=False)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    data_dir = args.inferencedata
    output_dir = args.output
    depth_dir = data_dir + "*" + ".png"
    depthset = glob.glob(depth_dir)
    #depthset = next(os.walk(path))[2]
    count = len(depthset)

    for i in range(count):
        print(depthset[i], " is now being inferred...")
        depth = skimage.io.imread(depthset[i])
        filename = os.path.splitext(os.path.basename(depthset[i]))[0]

        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                     mode='reflect', preserve_range=True)

        depth = torch.from_numpy(depth).float()
        depth.unsqueeze_(0)

        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)

        depth = depth.to(device).unsqueeze_(0)

        pred_depth = model(depth)

        output_depth = utils.color_label(torch.max(pred_depth, 1)[1] + 1)[0]

        output_dir_depth = os.path.join(output_dir, filename + ".png")

        imageio.imsave(output_dir_depth, output_depth.cpu().numpy().transpose((1, 2, 0)))

    print("Inference Completed!")


if __name__ == '__main__':
    inference()
