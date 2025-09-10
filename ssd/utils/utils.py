import torch
import torch.nn as nn
import math
import argparse
import time

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

eps_zero = 1e-15

def extract_layers(model, layers={}, pre_name='', get_conv=True, get_bn=False, get_gate=False):
    """
        Get all the model layers and the names of the layers
    Returns:
        layers: dict, the key is layer name and the value is the corresponding model layer
    """
    for name, layer in model.named_children():
        new_name = '.'.join([pre_name,name]) if pre_name != '' else name
        if len(list(layer.named_children())) > 0:
            extract_layers(layer, layers, new_name, get_conv, get_bn, get_gate)
        else:
            get_layer = False
            if get_conv and is_conv(layer):
                get_layer = True
            elif get_bn and is_bn(layer):
                get_layer = True
            elif get_gate and 'gate' in new_name:
                get_layer = True
            if get_layer:
                layers[new_name] = layer
    return layers

def is_conv(layer):
    conv_type = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
    isConv = False
    for ctp in conv_type:
        if isinstance(layer, ctp):
            isConv = True
            break
    return isConv


def is_bn(layer):
    bn_type = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    isbn = False
    for ctp in bn_type:
        if isinstance(layer, ctp):
            isbn = True
            break
    return isbn


def count_non_zero(input_tensor):
    """
        count the number of non-zero parameters in the input tensor
    """
    cmp = torch.ge(torch.abs(input_tensor), eps_zero)
    count = torch.sum(cmp.type(torch.float32))
    return int(count)


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, device=None, is_ddp=False):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    if device:
        true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    else:
        true_images = torch.LongTensor(true_images).cuda()
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    if device:
        det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    else:
        det_images = torch.LongTensor(det_images).cuda()
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)
    print('Total n_detections: {}'.format(det_scores.size(0)))
    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        if device:
            true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)
        else:
            true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).cuda()  # (n_class_objects)
            

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue
        print(c, n_class_detections)
        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        if device:
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        else:
            true_positives = torch.zeros((n_class_detections), dtype=torch.float).cuda() # (n_class_detections)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float).cuda()  # (n_class_detections)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = consistent_torch_max(overlaps.squeeze(0))  # (), () - scalars
            #max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        if device:
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        else:
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).cuda()  # (11)

        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    if is_ddp:
        # Must return torch tensors.
        return average_precisions
    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def count_non_zero_weights(layers, layer_weight_num=None, verbose=False):
    total_num = 0
    for name, layer in layers.items():
        if is_conv(layer) and (not 'predictor' in name):
            weights = layer.weight.data.view(-1)
            cur_num = count_non_zero(weights)
            total_num += cur_num
            if layer_weight_num is not None:
                layer_weight_num[name].append(cur_num)
            if verbose:
                print('{} has {} non-zero weights.'.format(name, cur_num))
    return total_num

def count_non_zero_neurons(layers, layer_neuron_num=None, neuron_norm=None, verbose=False):
    """
        count the total number of (conv) neurons
        if we are doing pruning, neuron_norm should be None.
    Returns:
        total_num: int
    """
    total_num = 0
    for name, layer in layers.items():
        if is_conv(layer) and (not 'predictor' in name):
            norm = layer.weight.data.view(layer.weight.data.size(0), -1).norm(dim=1)
            cur_num = count_non_zero(norm)
            total_num += cur_num
            if layer_neuron_num is not None:
                layer_neuron_num[name].append(cur_num)
            if neuron_norm is not None:
                avg_norm = float(norm.sum().item() / cur_num)
                print('layer {} AvgNorm: {}'.format(name, avg_norm))
                neuron_norm[name].append(norm.detach().cpu().numpy().squeeze())
            if verbose:
                print('{} has {} neurons.'.format(name, cur_num))
    return total_num


def count_all_weights(layers):
    total_num = 0
    for name, layer in layers.items():
        if is_conv(layer) and (not 'predictor' in name):
            norm = layer.weight.data.view(-1)
            cur_num = norm.size(0)
            total_num += cur_num
    return total_num


def count_all_neurons(layers):
    total_num = 0
    for name, layer in layers.items():
        if is_conv(layer) and (not 'predictor' in name):
            norm = layer.weight.data.view(layer.weight.data.size(0), -1).norm(dim=1)
            cur_num = norm.size(0)
            total_num += cur_num
    return total_num


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def adjust_learning_rate2(optimizer, value):
    """
    Modify learning rate to become a specific value.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param value: value that learning rate becomes.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = value
    #print("MODIFYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def save_checkpoint(epoch, model, optimizer, filename):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.curr_time = self.start_time

    def start(self):
        self.start_time = time.time()
        self.curr_time = self.start_time

    def lap_time(self):
        curr_time = time.time()
        elapsed_time = curr_time - self.curr_time
        self.curr_time = curr_time
        return elapsed_time

    def total_time(self, lap=True):
        curr_time = time.time()
        total_time = curr_time - self.start_time
        if lap:
            self.curr_time = curr_time
        return total_time

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                

# Takes in 0-dimensional tensor, and outputs leftmost argmax index.
def consistent_torch_max(input_tensor, leftmost=True):
    max_value = torch.max(input_tensor)
    tmp = torch.arange(input_tensor.size(0))
    if leftmost:
        return max_value, torch.min(tmp[input_tensor == max_value])
    else:
        return max_value, torch.max(tmp[input_tensor == max_value])
