from __future__ import division

import sys
import os

from projects.object_detection.modeling.model.models import *
from projects.object_detection.modeling.utils.util import *
from projects.object_detection.modeling.utils.datasets import *
from projects.object_detection.modeling.utils.transforms import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def get_detection_df(image_folder="data/samples", model_def="config/yolov3.cfg",
                     weights_path="weights/yolov3.weights", class_path="data/coco.names",
                     conf_thres=0.8, nms_thres=0.4,
                     batch_size=1, n_cpu=0, img_size=416):
    os.chdir(os.path.dirname(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output_dir = os.path.join(os.path.dirname(image_folder), image_folder.split("/")[-1] + "2")
    # if not os.path.isdir(output_dir):
    #     os.mkdir(output_dir)

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path) # Load darknet weights
    else:
        model.load_state_dict(torch.load(weights_path)) # Load checkpoint weights

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(image_folder, transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

    classes = load_classes(class_path)  # Extracts class labels from file

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # print("Performing object detection")

    detection_info_list = []
    file = ""
    label = 'none'
    confidence = 0
    topleft_x = 0
    topleft_y = 0
    bottomright_x = 0
    bottomright_y = 0
    width = 0
    height = 0

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

        if detections[0] is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                if int(cls_pred) == 16: #dog
                    file = img_paths[0].split("/")[-1]
                    label = classes[int(cls_pred)]
                    confidence = cls_conf.item()
                    topleft_x = x1.item()
                    topleft_y = y1.item()
                    bottomright_x = x2.item()
                    bottomright_y = y2.item()
                    width = (x2 - x1).item()
                    height = (y2 - y1).item()
                    # print(file, label, confidence, topleft_x, topleft_y, bottomright_x, bottomright_y, width, height)
                else: file = img_paths[0].split("/")[-1]
        else: ##이전 프레임의 정보를 넣기
            file = img_paths[0].split("/")[-1]
            label = 'none'

        detection_info = [file, label, confidence, topleft_x, topleft_y, bottomright_x, bottomright_y, width,height]
        detection_info_list.append(detection_info)

    column_list = ["file", "label", "confidence", "topleft_x", "topleft_y", "bottomright_x", "bottomright_y",
                   "width", "height"]
    detection_df = pd.DataFrame(detection_info_list, columns=column_list)
    frame_num = len(dataloader)

    ##############################################################################
    # # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    #
    # # print("\nSaving images:")
    # # Iterate through images and save plot of detections
    # for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    #
    #     # print("(%d) Image: '%s'" % (img_i, path))
    #
    #     # Create plot
    #     img = np.array(Image.open(path))
    #     plt.figure()
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img)
    #
    #     # Draw bounding boxes and labels of detections
    #     if detections is not None:
    #         # Rescale boxes to original image
    #         detections = rescale_boxes(detections, 416, img.shape[:2])
    #         unique_labels = detections[:, -1].cpu().unique()
    #         n_cls_preds = len(unique_labels)
    #         bbox_colors = random.sample(colors, n_cls_preds)
    #         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
    #             # print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
    #
    #             box_w = x2 - x1
    #             box_h = y2 - y1
    #
    #             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #             # Create a Rectangle patch
    #             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    #             # Add the bbox to the plot
    #             ax.add_patch(bbox)
    #             # Add label
    #             plt.text(
    #                 x1,
    #                 y1,
    #                 s=classes[int(cls_pred)],
    #                 color="white",
    #                 verticalalignment="top",
    #                 bbox={"color": color, "pad": 0},
    #             )
    #
    #     # Save generated image with detections
    #     plt.axis("off")
    #     plt.gca().xaxis.set_major_locator(NullLocator())
    #     plt.gca().yaxis.set_major_locator(NullLocator())
    #     filename = os.path.basename(path).split(".")[0]
    #     output_path = os.path.join(output_dir, f"{filename}.jpg")
    #     plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    #     plt.close()


    return detection_df, frame_num

if __name__=='__main__':
    pass
    # frame_dir = "/home/petpeotalk/Desktop/dogibogi/dogibogi-object-detection/dog_video/33204192_4e42f7a978_1_╢┘▒Γ_out"
    # detection_df = get_detection_df(image_folder=frame_dir)
