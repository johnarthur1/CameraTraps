r"""Train an EfficientNet classifier.

Currently implementation of multi-label multi-class classification is
non-functional.

During training, start tensorboard from within the classification/ directory:
    tensorboard --logdir run

Example usage:
    python train_classifier.py \
        run_idfg/classification_ds.csv \
        run_idfg/splits.json \
        /ssd/crops \
        -m "efficientnet-b0" --pretrained --finetune \
        --epochs 50 --batch-size 256 --num-workers 8 --seed 123
"""
import argparse
import json
import os
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils import data
import tqdm

from classification import efficientnet, train_classifier


def main(logdir: str, ckpt_name: str):
    """Main function."""
    with open(os.path.join(logdir, 'params.json'), 'r') as f:
        params = json.load(f)
    model_name = params['model_name']

    loaders, idx_to_label = train_classifier.create_dataloaders(
        classification_dataset_csv_path=params['classification_dataset_csv_path'],  # pylint: disable=line-too-long
        splits_json_path=params['splits_json_path'],
        cropped_images_dir=params['cropped_images_dir'],
        img_size=efficientnet.EfficientNet.get_image_size(params['model_name']),
        multilabel=params['multilabel'],
        label_weighted=True,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        augment_train=False)
    with open(os.path.join(logdir, 'label_index.json'), 'r') as f:
        saved_idx_to_label = json.load(f)
    # convert string keys to integers (JSON only permits string keys)
    saved_idx_to_label = {int(k): v for k, v in saved_idx_to_label.items()}
    for k in set(saved_idx_to_label.keys()) | set(idx_to_label.keys()):
        assert idx_to_label[k] == saved_idx_to_label[k]

    # create model
    num_classes = len(idx_to_label)
    model = efficientnet.EfficientNet.from_name(
        model_name, num_classes=num_classes)

    # load model weights from checkpoint
    ckpt_path = os.path.join(logdir, ckpt_name)
    print(f'Loading model from {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])

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

    # define loss function (criterion)
    criterion: torch.nn.Module
    if params['multilabel']:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    split_to_metrics = {}
    split_to_label_stats = {}
    for split, loader in loaders.items():
        print(split)
        split_to_metrics[split], split_to_label_stats[split] = test_epoch(
            model, loader=loader, weighted=True, device=device,
            label_ids=idx_to_label.keys(), criterion=criterion)

    metrics_sr = pd.concat(split_to_metrics, names=['split', 'metric'])
    metrics_sr.to_csv(os.path.join(logdir, 'overall_metrics.csv'))

    label_stats_df = pd.concat(
        split_to_label_stats, names=['split', 'label_id']).reset_index()
    label_stats_df['label'] = label_stats_df['label_id'].map(
        idx_to_label.__getitem__)
    label_stats_df.to_csv(os.path.join(logdir, 'label_stats.csv'), index=False)

    # calculate per-label statistics
    # label_stats_df.groupby('split').apply(per_label_acc)


def per_label_acc(df: pd.DataFrame):
    """df contains columns ['tp', 'fn']"""
    return df['tp'] / (df['tp'] + df['fn'])


def test_epoch(model: torch.nn.Module,
               loader: data.DataLoader,
               weighted: bool,
               device: torch.device,
               label_ids: Iterable[int],
               top: Sequence[int] = (1, 3),
               criterion: Optional[torch.nn.Module] = None
               ) -> Tuple[pd.Series, pd.DataFrame]:
    """Runs for 1 epoch.

    Args:
        model: torch.nn.Module
        loader: torch.utils.data.DataLoader
        device: torch.device
        top: tuple of int, list of values of k for calculating top-K accuracy
        criterion: optional loss function, calculates the mean loss over a batch
        label_ids: list of int

    Returns:
        metrics: pd.Series, index includes:
            'loss': float, mean per-example loss over entire epoch,
                only included if criterion is not None
            'acc_top{k}': float, accuracy@k over the entire epoch
        label_stats: pd.DataFrame, each row represents a label,
            columns are ['tp', 'fp', 'fn'], values are counts
    """
    # set dropout and BN layers to eval mode
    model.eval()

    if criterion is not None:
        losses = train_classifier.AverageMeter()
    accuracies = [train_classifier.AverageMeter() for _ in top]  # acc@k

    label_stats = pd.DataFrame(
        data=0, columns=['tp', 'fp', 'fn'], index=sorted(label_ids))

    tqdm_loader = tqdm.tqdm(loader)
    with torch.no_grad():
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

            for logits, target in zip(outputs, targets):
                ypred = logits.argmax().item()
                y = target.item()
                if ypred == y:
                    label_stats.loc[y, 'tp'] += 1
                else:
                    label_stats.loc[y, 'fn'] += 1
                    label_stats.loc[ypred, 'fp'] += 1

            top_correct = train_classifier.correct(
                outputs, targets, weights=weights, top=top)
            for acc, count, k in zip(accuracies, top_correct, top):
                acc.update(count * (100. / batch_size), n=batch_size)
                desc.append(f'Acc@{k} {acc.val:.3f} ({acc.avg:.3f})')
            tqdm_loader.set_description(' '.join(desc))

    metrics = {}
    if criterion is not None:
        metrics['loss'] = losses.avg
    for k, acc in zip(top, accuracies):
        metrics[f'acc_top{k}'] = acc.avg
    return pd.Series(metrics), label_stats



def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained model.')
    parser.add_argument(
        'logdir',
        help='path to logdir')
    parser.add_argument(
        'ckpt_name',
        help='name of checkpoint file from the logdir')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(logdir=args.logdir, ckpt_name=args.ckpt_name)
