import torch
import torch.nn.functional as F
import math
import torchvision
from utils.utils import *

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

                
def create_VOC_prior_boxes(is_SSD300=True):
    """
    Create the 8732/24564 prior (default) boxes for the SSD300/SSD512, as defined in the paper.

    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4) or (24564, 4)
    """
    if is_SSD300:
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}
    else:
        fmap_dims = {'conv4_3': 64,
                     'conv7': 32,
                     'conv8_2': 16,
                     'conv9_2': 8,
                     'conv10_2': 4,
                     'conv11_2': 2,
                     'conv12_2': 1}

        obj_scales = {'conv4_3': 0.07,
                      'conv7': 0.15,
                      'conv8_2': 0.30,
                      'conv9_2': 0.45,
                      'conv10_2': 0.60,
                      'conv11_2': 0.75,
                      'conv12_2': 0.90}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 3., 0.5, .333],
                         'conv11_2': [1., 2., 0.5],
                         'conv12_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * math.sqrt(ratio), obj_scales[fmap] / math.sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).cuda()  # (8732, 4) or (24564, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4) or (24564, 4)

    return prior_boxes


def detect_objects(priors_cxcy, n_classes, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
    """
    Decipher the 8732/24564 locations and class scores (output of ths SSD300/SSD512) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param predicted_locs: predicted locations/boxes w.r.t the 8732/24564 prior boxes, a tensor of dimensions (N, 8732/24564, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732/24564, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732/24564, n_classes)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732/24564, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732/24564)

        # Check for each class
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)
            score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732/24564
            class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)

            # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
            # 1 implies suppress, 0 implies don't suppress
            suppress = torch.zeros((n_above_min_score), dtype=torch.bool).cuda()  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                overlap_above_max = (overlap[box] > max_overlap)
                suppress = torch.max(suppress, overlap_above_max)

                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[~suppress])
            image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).cuda())
            image_scores.append(class_scores[~suppress])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).cuda())
            image_labels.append(torch.LongTensor([0]).cuda())
            image_scores.append(torch.FloatTensor([0.]).cuda())

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


# This version uses torchvision.ops.nms and is faster than the CPU version above.
def detect_objects2(priors_cxcy, n_classes, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
    """
    Decipher the 8732/24564 locations and class scores (output of ths SSD300/SSD512) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param predicted_locs: predicted locations/boxes w.r.t the 8732/24564 prior boxes, a tensor of dimensions (N, 8732/24564, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732/24564, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732/24564, n_classes)

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732/24564, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732/24564)

        # Check for each class
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)
            score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732/24564
            class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
            
            kept_indices = torchvision.ops.nms(class_decoded_locs, class_scores, max_overlap)

            image_boxes.append(class_decoded_locs[kept_indices])
            image_labels.append(torch.LongTensor(torch.numel(kept_indices) * [c]).cuda())
            image_scores.append(class_scores[kept_indices])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).cuda())
            image_labels.append(torch.LongTensor([0]).cuda())
            image_scores.append(torch.FloatTensor([0.]).cuda())

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size