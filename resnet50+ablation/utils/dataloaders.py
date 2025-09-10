import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

def expand(num_classes, dtype, tensor):
    e = torch.zeros(tensor.size(0), num_classes, dtype=dtype, device=torch.device('cuda'))
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e

class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, fp16, one_hot):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        if fp16:
            mean = mean.half()
            std = std.half()

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(async=True)
                next_target = next_target.cuda(async=True)
                if fp16:
                    next_input = next_input.half()
                    if one_hot:
                        next_target = expand(num_classes, torch.half, next_target)
                else:
                    next_input = next_input.float()
                    if one_hot:
                        next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, num_classes, fp16, one_hot):
        self.dataloader = dataloader
        self.fp16 = fp16
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader, self.num_classes, self.fp16, self.one_hot)

def get_pytorch_train_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False, dataset='ImageNet'):
    drop_last = True
    if dataset == 'ImageNet':
        traindir = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ]))
        drop_last = True
    elif dataset == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            ])
        train_dataset = torchvision.datasets.CIFAR10(
                root=data_path, train=True,
                download=True, transform=train_transform)
        drop_last = False
    else:
        raise NotImplementedError

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=drop_last)

    return PrefetchedWrapper(train_loader, num_classes, fp16, one_hot), len(train_loader)

def get_pytorch_val_loader(data_path, batch_size, num_classes, one_hot, workers=5, _worker_init_fn=None, fp16=False, dataset='ImageNet'):
    if dataset == 'ImageNet':
        valdir = os.path.join(data_path, 'val')
        val_dataset = datasets.ImageFolder(
                valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    ]))
    elif dataset == 'CIFAR10':
        val_dataset = torchvision.datasets.CIFAR10(
                root=data_path, train=False, download=True)
    else:
        raise NotImplementedError

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader, num_classes, fp16, one_hot), len(val_loader)

