r"""Train an EfficientNet classifier.

Currently implementation of multi-label multi-class classification is
non-functional.

During training, start tensorboard from within the classification/ directory:
    tensorboard --logdir run

TODO:
- plot learning rate in tensorboard
- run evaluate_mode.py at the end of training
- stop if val_loss and val_acc have not been increasing for 10 epochs
- investigate class weights
- verify that finetuning is really only changing the final-layer weights

Example usage:
    python train_classifier.py \
        run_idfg/classification_ds.csv \
        run_idfg/splits.json \
        /ssd/crops_sq \
        -m "efficientnet-b0" --pretrained --finetune --label-weighted \
        --epochs 50 --batch-size 512 --lr 1e-4 \
        --num-workers 12 --seed 123
"""
import argparse
from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import torch
from torch.utils import data, tensorboard
import torchvision
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import tqdm

from classification import efficientnet


# use accimage backend, which is faster than Pillow and Pillow-SIMD
torchvision.set_image_backend('accimage')

_NORMALIZE_TRANSFORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SimpleDataset(data.Dataset):
    """A simple dataset that simply returns images and labels."""

    def __init__(self,
                 img_paths: Sequence[str],
                 labels: Sequence[Any],
                 sample_weights: Optional[Sequence[float]] = None,
                 img_base_dir: str = '',
                 transform: Optional[Callable[[PIL.Image.Image], Any]] = None,
                 target_transform: Optional[Callable[[Any], Any]] = None):
        """Creates a SimpleDataset."""
        self.img_paths = img_paths
        self.labels = labels
        self.sample_weights = sample_weights
        self.img_base_dir = img_base_dir
        self.transform = transform
        self.target_transform = target_transform

        self.len = len(img_paths)
        assert len(labels) == self.len
        if sample_weights is not None:
            assert len(sample_weights) == self.len

    def __getitem__(self, index) -> Tuple[Any, ...]:
        """
        Args:
            index: int

        Returns: tuple, (sample, target) or (sample, target, sample_weight)
        """
        img_path = os.path.join(self.img_base_dir, self.img_paths[index])
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.sample_weights is not None:
            return img, target, self.sample_weights[index]
        return img, target

    def __len__(self) -> int:
        return self.len


def create_dataloaders(classification_dataset_csv_path: str,
                       splits_json_path: str,
                       cropped_images_dir: str,
                       img_size: int,
                       multilabel: bool,
                       label_weighted: bool,
                       batch_size: int,
                       num_workers: int,
                       augment_train: bool
                       ) -> Tuple[Dict[str, data.DataLoader], Dict[int, str]]:
    """
    Args:
        classification_dataset_csv_path: str, path to CSV file with columns
            ['dataset', 'location', 'label'], where label is a comma-delimited
            list of labels
        splits_json_path: str, path to JSON file
        augment_train: bool, whether to shuffle/augment the training set

    Returns:
        datasets: dict, maps split to DataLoader
        idx_to_label: dict, maps label index to label name
    """
    # read in dataset CSV and create merged (dataset, location) col
    df = pd.read_csv(classification_dataset_csv_path, index_col=False)
    df['dataset_location'] = list(zip(df['dataset'], df['location']))

    # create mappings from labels to int
    if multilabel:
        df['label'] = df['label'].map(lambda x: x.split(','))
        all_labels = set(df['label'].explode())
        # look into sklearn.preprocessing.MultiLabelBinarizer
    else:
        assert not any(df['label'].str.contains(','))
        all_labels = set(df['label'])

    idx_to_label = dict(enumerate(sorted(all_labels)))
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}

    # map label to label_index
    if multilabel:
        df['label_index'] = df['label'].map(
            lambda labellist: tuple(sorted(label_to_idx[y] for y in labellist)))
    else:
        df['label_index'] = df['label'].map(label_to_idx.__getitem__)

    # load the splits and assert that there are no overlaps in locs
    with open(splits_json_path, 'r') as f:
        split_to_locs = json.load(f)
    split_to_locs = {
        split: set(tuple(loc) for loc in locs)
        for split, locs in split_to_locs.items()
    }
    assert split_to_locs['train'].isdisjoint(split_to_locs['val'])
    assert split_to_locs['train'].isdisjoint(split_to_locs['test'])
    assert split_to_locs['val'].isdisjoint(split_to_locs['test'])

    # define the transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE_TRANSFORM
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        _NORMALIZE_TRANSFORM
    ])

    if label_weighted:
        # TODO: handle multilabel case
        train_df = df[df['dataset_location'].isin(split_to_locs['train'])]
        label_weights = (1 / train_df['label_index'].value_counts()).to_dict()

    dataloaders = {}
    for split, locs in split_to_locs.items():
        is_train = (split == 'train') and augment_train
        split_df = df[df['dataset_location'].isin(locs)]

        sample_weights = None
        if label_weighted:
            # TODO: handle multilabel case
            # sample weights must sum to the # of examples in this split
            sample_weights = split_df['label_index'].map(
                label_weights.__getitem__)
            sample_weights *= len(split_df) / sample_weights.sum()
            sample_weights = sample_weights.tolist()

        dataset = SimpleDataset(
            img_paths=split_df['path'].tolist(),
            labels=split_df['label_index'].tolist(),
            sample_weights=sample_weights,
            img_base_dir=cropped_images_dir,
            transform=train_transform if is_train else test_transform)
        dataloaders[split] = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=True)

    return dataloaders, idx_to_label


def prefix_all_keys(d: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Returns a new dict where the keys are prefixed by <prefix>."""
    return {f'{prefix}{k}': v for k, v in d.items()}


def log_metrics(writer: tensorboard.SummaryWriter, metrics: Dict[str, float],
                epoch: int, prefix: str = '') -> None:
    """Logs metrics to TensorBoard."""
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}{metric}', value, epoch)


def main(classification_dataset_csv_path: str,
         splits_json_path: str,
         cropped_images_dir: str,
         multilabel: bool,
         model_name: str,
         pretrained: bool,
         finetune: bool,
         label_weighted: bool,
         epochs: int,
         batch_size: int,
         lr: float,
         num_workers: int,
         seed: Optional[int] = None):
    """Main function."""
    # set seed
    seed = np.random.randint(10_000) if seed is None else seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # create logdir and save params
    params = dict(locals())  # make a copy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # '20200722_110816'
    logdir = os.path.join('run', timestamp)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=1)

    writer = tensorboard.SummaryWriter(logdir)

    # create dataloaders and log the index_to_label mapping
    loaders, idx_to_label = create_dataloaders(
        classification_dataset_csv_path=classification_dataset_csv_path,
        splits_json_path=splits_json_path,
        cropped_images_dir=cropped_images_dir,
        img_size=efficientnet.EfficientNet.get_image_size(model_name),
        multilabel=multilabel,
        label_weighted=label_weighted,
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=True)
    with open(os.path.join(logdir, 'label_index.json'), 'w') as f:
        # Note: JSON always saves keys as strings!
        json.dump(idx_to_label, f, indent=1)

    # create model
    num_classes = len(idx_to_label)
    if pretrained:
        model = efficientnet.EfficientNet.from_pretrained(
            model_name, num_classes=num_classes)
    else:
        model = efficientnet.EfficientNet.from_name(
            model_name, num_classes=num_classes)
    if finetune:
        # set all parameters to not require gradients except final FC layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model._fc.parameters():  # pylint: disable=protected-access
            param.requires_grad = True

    # detect GPU, use all if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = torch.device('cpu')
    model.to(device)  # in-place

    # define loss function (criterion) and optimizer
    criterion: torch.nn.Module
    if multilabel:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # using EfficientNet training defaults
    # - batch norm momentum: 0.99
    # - optimizer: RMSProp, decay 0.9 and momentum 0.9
    # - epochs: 350
    # - learning rate: 0.256, decays by 0.97 every 2.4 epochs
    # - weight decay: 1e-5
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr, alpha=0.9, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.97 ** (1 / 2.4))

    best_epoch_metrics: Dict[str, float] = {}
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch)

        print('- train:')
        train_metrics = run_epoch(
            model, loader=loaders['train'], weighted=label_weighted,
            device=device, criterion=criterion, finetune=finetune,
            optimizer=optimizer)
        train_metrics = prefix_all_keys(train_metrics, prefix='train/')
        log_metrics(writer, train_metrics, epoch)

        print('- val:')
        val_metrics = run_epoch(
            model, loader=loaders['val'], weighted=label_weighted,
            device=device, criterion=criterion)
        val_metrics = prefix_all_keys(val_metrics, prefix='val/')
        log_metrics(writer, val_metrics, epoch)

        lr_scheduler.step()  # decrease the learning rate

        if val_metrics['val/acc_top1'] > best_epoch_metrics.get('val/acc_top1', 0):  # pylint: disable=line-too-long
            filename = os.path.join(logdir, f'ckpt_{epoch}.pt')
            print(f'New best model! Saving checkpoint to {filename}')
            state = {
                'epoch': epoch,
                'model': getattr(model, 'module', model).state_dict(),
                'val/acc': val_metrics['val/acc_top1'],
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, filename)
            best_epoch_metrics.update(train_metrics)
            best_epoch_metrics.update(val_metrics)
            best_epoch_metrics['epoch'] = epoch

    hparams_dict = {
        'model_name': model_name,
        'multilabel': multilabel,
        'finetune': finetune,
        'batch_size': batch_size,
        'epochs': epochs
    }
    metric_dict = prefix_all_keys(best_epoch_metrics, prefix='hparam/')
    writer.add_hparams(hparam_dict=hparams_dict, metric_dict=metric_dict)
    writer.close()


def correct(outputs: torch.Tensor, targets: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            top: Sequence[int] = (1,)) -> List[float]:
    """
    Args:
        outputs: torch.Tensor, shape [N, num_classes],
            either logits (pre-softmax) or probabilities
        targets: torch.Tensor, shape [N]
        weights: optional torch.Tensor, shape [N]
        top: tuple of int, list of values of k for calculating top-K accuracy

    Returns: list of float, same length as top, (weighted) # of correct
        predictions @ each k
    """
    with torch.no_grad():
        # preds and targets both have shape [N, k]
        _, preds = outputs.topk(k=max(top), dim=1, largest=True, sorted=True)
        targets = targets.view(-1, 1).expand_as(preds)

        corrects = preds.eq(targets).cumsum(dim=1)  # shape [N, k]
        if weights is None:
            corrects = corrects.sum(dim=0)  # shape [k]
        else:
            corrects = weights.matmul(corrects.to(weights.dtype))  # shape [k]
        tops = list(map(lambda k: corrects[k - 1].item(), top))
    return tops


def run_epoch(model: torch.nn.Module,
              loader: data.DataLoader,
              weighted: bool,
              device: torch.device,
              top: Sequence[int] = (1, 3),
              criterion: Optional[torch.nn.Module] = None,
              finetune: bool = False,
              optimizer: Optional[torch.optim.Optimizer] = None,
              ) -> Dict[str, float]:
    """Runs for 1 epoch.

    Args:
        model: torch.nn.Module
        loader: torch.utils.data.DataLoader
        device: torch.device
        top: tuple of int, list of values of k for calculating top-K accuracy
        criterion: optional loss function, calculates the mean loss over a batch
        finetune: bool, if true sets model's dropout and BN layers to eval mode
        optimizer: optional optimizer

    Returns:
        metrics: dict, metrics from epoch, contains keys:
            'loss': float, mean per-example loss over entire epoch,
                only included if criterion is not None
            'acc_top{k}': float, accuracy@k over the entire epoch
    """
    if optimizer is not None:
        assert criterion is not None

    # if evaluating or finetuning, set dropout and BN layers to eval mode
    model.train(optimizer is not None and not finetune)

    if criterion is not None:
        losses = AverageMeter()
    accuracies = [AverageMeter() for _ in top]  # acc@k

    tqdm_loader = tqdm.tqdm(loader)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm_loader:
            if weighted:
                inputs, targets, weights = batch
                weights = weights.to(device, non_blocking=True)
            else:
                inputs, targets = batch
                weights = None

            batch_size = targets.size(0)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            desc = []
            if criterion is not None:
                if weighted:
                    loss = criterion(outputs, targets) * weights
                else:
                    loss = criterion(outputs, targets)
                loss = loss.mean()
                losses.update(loss.item(), n=batch_size)
                desc.append(f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top_correct = correct(outputs, targets, weights=weights, top=top)
            for acc, count, k in zip(accuracies, top_correct, top):
                acc.update(count * (100. / batch_size), n=batch_size)
                desc.append(f'Acc@{k} {acc.val:.3f} ({acc.avg:.3f})')
            tqdm_loader.set_description(' '.join(desc))

    metrics = {}
    if criterion is not None:
        metrics['loss'] = losses.avg
    for k, acc in zip(top, accuracies):
        metrics[f'acc_top{k}'] = acc.avg
    return metrics


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Trains classifier.')
    parser.add_argument(
        'classification_dataset_csv',
        help='path to CSV file crop paths and classification info')
    parser.add_argument(
        'splits_json',
        help='path to JSON file with splits information')
    parser.add_argument(
        'cropped_images_dir',
        help='path to local directory where image crops are saved')
    parser.add_argument(
        '--multilabel', action='store_true',
        help='for multi-label, multi-class classification')
    parser.add_argument(
        '-m', '--model-name', default='efficientnet-b0',
        choices=efficientnet.VALID_MODELS,
        help='which EfficientNet model (default: efficientnet-b0)')
    parser.add_argument(
        '--pretrained', action='store_true',
        help='start with pretrained model')
    parser.add_argument(
        '--finetune', action='store_true',
        help='only fine tune the final fully-connected layer')
    parser.add_argument(
        '--label-weighted', action='store_true',
        help='weight training samples to balance labels')
    parser.add_argument(
        '--epochs', type=int, default=0,
        help='number of epochs for training (default: 0, eval only)')
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='batch size for both training and eval')
    parser.add_argument(
        '--lr', type=float, default=None,
        help='initial learning rate, defaults to (0.016 * batch_size / 256)')
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='number of workers for data loading')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.lr is None:
        args.lr = 0.016 * args.batch_size / 256  # based on TF models repo
    main(classification_dataset_csv_path=args.classification_dataset_csv,
         splits_json_path=args.splits_json,
         cropped_images_dir=args.cropped_images_dir,
         multilabel=args.multilabel,
         model_name=args.model_name,
         pretrained=args.pretrained,
         finetune=args.finetune,
         label_weighted=args.label_weighted,
         epochs=args.epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         num_workers=args.num_workers,
         seed=args.seed)
