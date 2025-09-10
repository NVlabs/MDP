from utils.utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from ssd_models.models import SSD
import argparse, glob, os
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import time
from collections import OrderedDict
from utils.ssd_utils import *
from masking import *
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!

model_names = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', \
               'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}
n_classes = len(label_map)

def add_masking_hooks(layers, layer_bn):
    for layer_name, layer in layers.items():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
            add_attr_masking(layer, 'weight', ParameterMaskingType.Hard, True)
            if hasattr(layer, "bias") and layer.bias is not None:
                add_attr_masking(layer, 'bias', ParameterMaskingType.Hard, True)
            if layer_bn is not None and layer_name in layer_bn:
                bn_layer = layers[layer_bn[layer_name]]
                add_attr_masking(bn_layer, 'weight', ParameterMaskingType.Hard, True)
                add_attr_masking(bn_layer, 'bias', ParameterMaskingType.Hard, True)


def evaluate(gpu, args):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """
    batch_size = args.batch_size
    workers = args.workers
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.gpus, rank=gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    if args.random_noise:
        torch.manual_seed(args.random_seed)

    def sync_tensor_list(output_tensor_list, input_tensor):
        torch.cuda.synchronize()
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda()
        dist.all_gather(output_tensor_list, input_tensor)
        torch.cuda.synchronize()

    # Preprocess ground truths.
    # Lists to store ground truth, labels, scores
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    if args.model=='SSD300':
        input_size = (300, 300)
    else:
        input_size = (512, 512)
    # Load test data
    test_dataset = PascalVOCDataset(args.data_dir,
                                    split='test',
                                    keep_difficult=keep_difficult, input_size=input_size)
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.gpus, rank=gpu, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size//args.gpus, shuffle=False, sampler=test_sampler,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers//args.gpus, pin_memory=True)


    with torch.no_grad():
        # Batches
        for i, (_, boxes, labels, difficulties) in enumerate(test_loader):
            boxes = [b.cuda() for b in boxes]
            labels = [l.cuda() for l in labels]
            difficulties = [d.cuda() for d in difficulties]
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        batch_count = len(true_labels)
        batch_count = torch.tensor(batch_count)
        batch_count_sizes = [torch.tensor(0).cuda() for _ in range(args.gpus)]
        sync_tensor_list(batch_count_sizes, batch_count)

        cum_sizes = [0]
        for size in batch_count_sizes:
            cum_sizes.append(cum_sizes[-1] + size.item())

        # Sync all ground truths.
        true_images = list()
        for i in range(batch_count):
            true_images.extend([cum_sizes[gpu] + i] * true_labels[i].size(0))
        true_images = torch.LongTensor(true_images).cuda()  # (n_objects), n_objects is the total no. of objects across all images

        true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
        true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
        true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

        assert true_difficulties.size(0) == true_boxes.size(0) == true_labels.size(0)

        # Sync true_images, true_boxes, true_labels, true_difficulties
        n_objects = true_labels.size(0)
        n_objects = torch.tensor(n_objects)

        n_objects_sizes = [torch.tensor(0).cuda() for _ in range(args.gpus)]
        sync_tensor_list(n_objects_sizes, n_objects)
        max_size = 0
        sizes = []
        for size in n_objects_sizes:
            max_size = max(max_size, size.item())
            sizes.append(size.item())
        
        true_images_padded = torch.cat([true_images, torch.zeros(max_size-true_images.size(0), dtype=torch.long).cuda()])
        true_images_list = [torch.zeros(max_size, dtype=torch.long).cuda() for _ in range(args.gpus)]
        sync_tensor_list(true_images_list, true_images_padded)

        true_boxes_padded = torch.cat([true_boxes, torch.zeros((max_size-true_boxes.size(0), 4), dtype=torch.float).cuda()])
        true_boxes_list = [torch.zeros(max_size, 4, dtype=torch.float).cuda() for _ in range(args.gpus)]
        sync_tensor_list(true_boxes_list, true_boxes_padded)

        true_labels_padded = torch.cat([true_labels, torch.zeros(max_size-true_labels.size(0), dtype=torch.long).cuda()])
        true_labels_list = [torch.zeros(max_size, dtype=torch.long).cuda() for _ in range(args.gpus)]
        sync_tensor_list(true_labels_list, true_labels_padded)

        true_difficulties_padded = torch.cat([true_difficulties, torch.zeros(max_size-true_difficulties.size(0), dtype=torch.uint8).cuda()])
        true_difficulties_list = [torch.zeros(max_size, dtype=torch.uint8).cuda() for _ in range(args.gpus)]
        sync_tensor_list(true_difficulties_list, true_difficulties_padded)

        concat_true_images = [true_images_list[i][:sizes[i]] for i in range(args.gpus)]
        true_images = torch.cat(concat_true_images, dim=0)
        concat_true_boxes = [true_boxes_list[i][:sizes[i]][:] for i in range(args.gpus)]
        true_boxes = torch.cat(concat_true_boxes, dim=0)
        concat_true_labels = [true_labels_list[i][:sizes[i]] for i in range(args.gpus)]
        true_labels = torch.cat(concat_true_labels, dim=0)
        concat_true_difficulties = [true_difficulties_list[i][:sizes[i]] for i in range(args.gpus)]
        true_difficulties = torch.cat(concat_true_difficulties, dim=0)


    model = SSD(model=args.model, backbone=args.arch, n_classes=n_classes, pretrained=False, batch_norm=not args.disable_batch_norm, no_gating=not args.enable_gating)
    model = model.cuda(gpu)
    model = DDP(model, device_ids=[gpu])

    test_dataset = PascalVOCDataset(args.data_dir,
                                    split='test',
                                    keep_difficult=keep_difficult, only_image=True, random_noise=args.random_noise, random_std_scale_factor=args.random_std_scale_factor,
                                    input_size = input_size)
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.gpus, rank=gpu, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size//args.gpus, shuffle=False, sampler=test_sampler,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers//args.gpus, pin_memory=True)

    for checkpoint in tqdm(args.all_checkpoints, desc='Evaluating'):
        c_start = time.time()
        before_or_after = checkpoint.split('_')[-2]
        if before_or_after not in ['before', 'after']:
            before_or_after = ''
        curr_file_name = checkpoint.split('/')[-1]
        curr_file_name_without_extension = curr_file_name[:-4]
        checkpoint_epoch = int(curr_file_name_without_extension.split('_')[1])
        # Load model checkpoint that is to be evaluated
        checkpoint = torch.load(checkpoint)
        model_state_dict = OrderedDict()
        if 'model_state_dict' in checkpoint:
            for k, v in checkpoint['model_state_dict'].items():
                model_state_dict[k.replace("features_0", "f_0")] = v
        else:
            for k, v in checkpoint.items():
                model_state_dict[k.replace("features_0", "f_0")] = v

        # layers = extract_layers(model, get_conv=True, get_bn=True, get_gate=False)
        # config_reader = PruneConfigReader(gpu==0)
        # config_reader.set_prune_setting(args.reg_conf)
        # layer_structure = config_reader.get_layer_structure()

        for k, v in model_state_dict.items():
            if '_orig' in k:
                new_k = k.replace('_orig', '')
                # print(model_state_dict[k].shape)
                model_state_dict[k] = model_state_dict[k] * model_state_dict[f"{new_k}_mask"]
                # print(model_state_dict[k].shape)

        model_state_dict = {k.replace('_orig', ''): v for k, v in model_state_dict.items() if "mask" not in k}
        for key in model_state_dict:
            if "_mask" in key:
                del model_state_dict[key]

        model.load_state_dict(model_state_dict)
        # Switch to eval mode
        model.eval()
        
        # Lists to store detected boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()

        with torch.no_grad():
            # Batches
            for i, (images) in enumerate(test_loader):
                images = images.cuda()  # (N, 3, 300, 300)
                predicted_locs, predicted_scores = model(images)
                # Detect objects in SSD output
                det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects2(model.module.priors_cxcy, n_classes, predicted_locs, predicted_scores,
                                                                                                  min_score=0.01, max_overlap=0.45,
                                                                                                  top_k=200)
                # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                # Store this batch's results for mAP calculation
                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)

            # Store all detections in a single continuous tensor while keeping track of the image it is from
            det_images = list()
            for i in range(batch_count):
                det_images.extend([cum_sizes[gpu] + i] * det_labels[i].size(0))
            det_images = torch.LongTensor(det_images).cuda()  # (n_detections)

            det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
            det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
            det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

            # Sync all detections
            n_detections = det_boxes.size(0)
            n_detections = torch.tensor(n_detections, dtype=torch.long)

            n_detections_sizes = [torch.tensor(0, dtype=torch.long).cuda() for _ in range(args.gpus)]
            sync_tensor_list(n_detections_sizes, n_detections)
            max_size = 0
            sizes = []
            for size in n_detections_sizes:
                max_size = max(max_size, size.item())
                sizes.append(size.item())

            det_images_padded = torch.cat([det_images, torch.zeros(max_size-det_images.size(0), dtype=torch.long).cuda()])
            det_images_list = [torch.zeros(max_size, dtype=torch.long).cuda() for _ in range(args.gpus)]
            sync_tensor_list(det_images_list, det_images_padded)

            det_boxes_padded = torch.cat([det_boxes, torch.zeros((max_size-det_boxes.size(0), 4), dtype=torch.float).cuda()])
            det_boxes_list = [torch.zeros((max_size, 4), dtype=torch.float).cuda() for _ in range(args.gpus)]
            sync_tensor_list(det_boxes_list, det_boxes_padded)

            det_labels_padded = torch.cat([det_labels, torch.zeros(max_size-det_labels.size(0), dtype=torch.long).cuda()])
            det_labels_list = [torch.zeros(max_size, dtype=torch.long).cuda() for _ in range(args.gpus)]
            sync_tensor_list(det_labels_list, det_labels_padded)

            det_scores_padded = torch.cat([det_scores, torch.zeros(max_size-det_scores.size(0), dtype=torch.float).cuda()])
            det_scores_list = [torch.zeros(max_size, dtype=torch.float).cuda() for _ in range(args.gpus)]
            sync_tensor_list(det_scores_list, det_scores_padded)

            concat_det_images = [det_images_list[i][:sizes[i]] for i in range(args.gpus)]
            det_images = torch.cat(concat_det_images, dim=0)
            concat_det_boxes = [det_boxes_list[i][:sizes[i]][:] for i in range(args.gpus)]
            det_boxes = torch.cat(concat_det_boxes, dim=0)
            concat_det_labels = [det_labels_list[i][:sizes[i]] for i in range(args.gpus)]
            det_labels = torch.cat(concat_det_labels, dim=0)
            concat_det_scores = [det_scores_list[i][:sizes[i]] for i in range(args.gpus)]
            det_scores = torch.cat(concat_det_scores, dim=0)


            # Calculate APs for each class (except background)
            if gpu == 0:
                average_precisions = torch.zeros((n_classes - 1), dtype=torch.float).cuda()  # (n_classes - 1)

            for c in range(1, n_classes):

                true_class_images = true_images[true_labels == c]  # (n_class_objects)
                true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
                true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
                n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

                # Extract only detections with this class
                det_class_images = det_images[det_labels == c]  # (n_class_detections)
                det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
                det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
                n_class_detections = det_class_images.size(0)
                if n_class_detections == 0:
                    continue

                det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
                det_class_images = det_class_images[sort_ind]  # (n_class_detections)
                det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

                indices = [int(n_class_detections*i/args.gpus) for i in range(args.gpus)]
                indices.append(n_class_detections)
                sizes = [(indices[i+1] - indices[i]) for i in range(args.gpus)]
                max_n_class_detections = max(sizes)
                start_ind = indices[gpu]
                end_ind = indices[gpu+1]

                true_positives = torch.zeros((max_n_class_detections), dtype=torch.float).cuda() # (n_class_detections)
                false_positives = torch.zeros((max_n_class_detections), dtype=torch.float).cuda()  # (n_class_detections)
                original_ind_mapping = torch.zeros((max_n_class_detections), dtype=torch.long).cuda() # Mapping from d -> original_ind

                for d in range(start_ind, end_ind):
                    this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                    this_image = det_class_images[d]  # (), scalar

                    # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                    object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                    object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
                    # If no such object in this image, then the detection is a false positive
                    if object_boxes.size(0) == 0:
                        false_positives[d-start_ind] = 1
                        continue

                    # Find maximum overlap of this detection with objects in this image of this class
                    overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
                    max_overlap, ind = consistent_torch_max(overlaps.squeeze(0))  # (), () - scalars

                    # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                    
                    # If the maximum overlap is greater than the threshold of 0.5, it's a match
                    if max_overlap.item() > 0.5:
                        # If the object it matched with is 'difficult', ignore it
                        if object_difficulties[ind] == 0:
                            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]   # 0 <= original_ind < n_class_objects
                            # We need 'original_ind' to update 'true_class_boxes_detected'
                            original_ind_mapping[d-start_ind] = original_ind
                            
                            # If this object has already not been detected, it's a true positive
                            true_positives[d-start_ind] = 1
                    # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                    else:
                        false_positives[d-start_ind] = 1

                # All-gather true_positives and false_positives
                true_positives_list = [torch.zeros(max_n_class_detections).cuda() for _ in range(args.gpus)]
                #true_positives_padded = torch.cat([true_positives, torch.zeros((sum_of_sizes // 7)-n_class_detections, dtype=torch.float).cuda()])
                sync_tensor_list(true_positives_list, true_positives)

                false_positives_list = [torch.zeros(max_n_class_detections).cuda() for _ in range(args.gpus)]
                #false_positives_padded = torch.cat([false_positives, torch.zeros((sum_of_sizes // 7)-n_class_detections, dtype=torch.float).cuda()])
                sync_tensor_list(false_positives_list, false_positives)

                original_ind_mapping_list = [torch.zeros(max_n_class_detections, dtype=torch.long).cuda() for _ in range(args.gpus)]
                sync_tensor_list(original_ind_mapping_list, original_ind_mapping)

                if gpu == 0:
                    concat_true_positives = [true_positives_list[i][:sizes[i]] for i in range(args.gpus)]
                    true_positives = torch.cat(concat_true_positives, dim=0)
                    concat_false_positives = [false_positives_list[i][:sizes[i]] for i in range(args.gpus)]
                    false_positives = torch.cat(concat_false_positives, dim=0)
                    concat_original_ind_mapping = [original_ind_mapping_list[i][:sizes[i]] for i in range(args.gpus)]
                    original_ind_mapping = torch.cat(concat_original_ind_mapping, dim=0)
                    assert true_positives.size(0) == false_positives.size(0) == original_ind_mapping.size(0) == n_class_detections

                    final_true_class_boxes_detected = torch.zeros(true_class_difficulties.size(0), dtype=torch.uint8).cuda()
                    
                    for d in range(n_class_detections):
                        if true_positives[d] == 1:
                            original_ind = original_ind_mapping[d]
                            if final_true_class_boxes_detected[original_ind] == 1: # this object has now been accounted for, so must put this as false positive
                                true_positives[d] = 0
                                false_positives[d] = 1
                            else:
                                final_true_class_boxes_detected[original_ind] = 1 # this object has now been accounted for the first time.

                    cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
                    cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
                    cumul_precision = cumul_true_positives / (
                            cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
                    cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

                    # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
                    recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
                    precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).cuda()  # (11)

                    for i, t in enumerate(recall_thresholds):
                        recalls_above_t = cumul_recall >= t
                        if recalls_above_t.any():
                            precisions[i] = cumul_precision[recalls_above_t].max()
                        else:
                            precisions[i] = 0.
                    average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

            if gpu == 0:
                mean_average_precision = average_precisions.mean().item()

                # Keep class-wise average precisions in a dictionary
                average_precisions = {rev_label_map[c+1]: v for c, v in enumerate(average_precisions.tolist())}
                

                before_or_after_str = ''
                if before_or_after != '':
                    before_or_after_str = ' ({:6} pruning)'.format(before_or_after)
                with open(args.out_file, "a+") as f:
                    c_end = time.time()
                    f.write('Epoch {:0>3d}: mAP is {:.4f}. (Eval took {:.2f} seconds){}\n'.format(checkpoint_epoch, mean_average_precision, c_end-c_start, before_or_after_str))
                # Print AP for each class
                if args.print_stats:
                    pp.pprint(average_precisions)
                    print('\nEpoch {:0>3d}: mAP is {:.3f}. (Eval took {:.2f} seconds){}\n'.format(checkpoint_epoch, mean_average_precision, c_end-c_start, before_or_after_str))


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('--data-dir', default='/workspace/marvink/json',
                        help='Data directory containing json files.')
    parser.add_argument('--checkpoint-dir',
                        help='Checkpoint directory containing pth files and runs. This directory will also be the output file parent directory.')
    parser.add_argument('--preprocess-dir', default='/workspace/marvink/preprocessed_true_labels',
                        help='Directory containing the preprocessed true labels.')
    parser.add_argument('--print-stats', action='store_true',
                        help='Whether or not to print APs/mAP for every checkpoint.')
    parser.add_argument('--workers', default=16, type=int, metavar='N',
                        help='number of total data loading workers (default: 16)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='The number of examples in one training batch')
    parser.add_argument('-s', '--start-epoch', type=int,
                        help='The first epoch to evaluate.')
    parser.add_argument('-e', '--end-epoch', type=int,
                        help='The last epoch to evaluate.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet34)')
    parser.add_argument('--enable-gating', action='store_true')
    parser.add_argument('--disable-batch-norm', action='store_true',
                        help='Disable batch norm for additional layers.')
    parser.add_argument('--model', type=str, default='SSD300',
                        help='Which SSD model to use (default: SSD300)')
    parser.add_argument('--random-noise', action='store_true',
                        help='Adds random Gaussian noise to the images')
    parser.add_argument('--random-seed', default=7, type=int,
                        help='Set random seed, used for random noise generation.')
    parser.add_argument('--random-std-scale-factor', default=0.1, type=float,
                        help='Set the random std scale factor. A larger value is more random Gaussian noise.')
    args = parser.parse_args()  

    if not args.checkpoint_dir:
        all_result_dirs = [os.path.join('/result', dir) for dir in os.listdir('/result') if os.path.isdir(os.path.join('/result', dir))]
        args.checkpoint_dir = sorted(all_result_dirs)[-1]
    if not args.checkpoint_dir.endswith('VOC_checkpoints') and not args.checkpoint_dir.endswith('VOC_checkpoints/'):
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'VOC_checkpoints')
    args.all_checkpoints = sorted([os.path.join(args.checkpoint_dir, dir) for dir in os.listdir(args.checkpoint_dir) if dir.endswith('.pth')])
    
    if args.start_epoch or args.end_epoch:
        run_checkpoints = []
        for checkpoint in args.all_checkpoints:
            curr_file_name = checkpoint.split('/')[-1]
            curr_file_name_without_extension = curr_file_name.split('.')[0]
            checkpoint_epoch = int(curr_file_name_without_extension.split('_')[1])
            if args.start_epoch and checkpoint_epoch < args.start_epoch:
                continue
            else:
                if args.end_epoch and checkpoint_epoch >= args.end_epoch:
                    continue
                else:
                    run_checkpoints.append(checkpoint)
        args.all_checkpoints = run_checkpoints
    print('Number of checkpoints being evaluated: {}'.format(len(args.all_checkpoints)))

    args.out_file = os.path.join(os.path.dirname(args.checkpoint_dir), 'mAP_eval.txt')
    print(args.out_file)
    if not os.path.exists(args.out_file):
        with open(args.out_file, "w+") as f: 
            f.write("mAP for VOC2007 test set:\n") 
        
    print("Evaluating checkpoints in {}".format(args.checkpoint_dir))

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12367'
    mp.spawn(evaluate, nprocs=args.gpus, args=(args,))
    end = time.time()
    print("elapsed time in seconds: {}".format(end-start))
