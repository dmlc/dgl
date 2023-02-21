import argparse
import time
from functools import partial

import dgl

import numpy as np
import torch
import torch.optim as optim
import tqdm
from point_transformer import PartSegLoss, PointTransformerSeg
from ShapeNet import ShapeNet
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="")
parser.add_argument("--load-model-path", type=str, default="")
parser.add_argument("--save-model-path", type=str, default="")
parser.add_argument("--num-epochs", type=int, default=250)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--tensorboard", action="store_true")
parser.add_argument("--opt", type=str, default="adam")
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size


def collate(samples):
    graphs, cat = map(list, zip(*samples))
    return dgl.batch(graphs), cat


CustomDataLoader = partial(
    DataLoader,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


def train(net, opt, scheduler, train_loader, dev):
    category_list = sorted(list(shapenet.seg_classes.keys()))
    eye_mat = np.eye(16)
    net.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    start = time.time()
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label, cat in tq:
            num_examples = data.shape[0]
            data = data.to(dev, dtype=torch.float)
            label = label.to(dev, dtype=torch.long).view(-1)
            opt.zero_grad()
            cat_ind = [category_list.index(c) for c in cat]
            # An one-hot encoding for the object category
            cat_tensor = (
                torch.tensor(eye_mat[cat_ind])
                .to(dev, dtype=torch.float)
                .repeat(1, 2048)
            )
            cat_tensor = cat_tensor.view(num_examples, -1, 16)
            logits = net(data, cat_tensor).permute(0, 2, 1)
            loss = L(logits, label)
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            count += num_examples * 2048
            loss = loss.item()
            total_loss += loss
            num_batches += 1
            correct = (preds.view(-1) == label).sum().item()
            total_correct += correct

            AvgLoss = total_loss / num_batches
            AvgAcc = total_correct / count

            tq.set_postfix(
                {"AvgLoss": "%.5f" % AvgLoss, "AvgAcc": "%.5f" % AvgAcc}
            )
    scheduler.step()
    end = time.time()
    print(
        "[Train] AvgLoss: {:.5}, AvgAcc: {:.5}, Time: {:.5}s".format(
            total_loss / num_batches, total_correct / count, end - start
        )
    )
    return data, preds, AvgLoss, AvgAcc, end - start


def mIoU(preds, label, cat, cat_miou, seg_classes):
    for i in range(preds.shape[0]):
        shape_iou = 0
        n = len(seg_classes[cat[i]])
        for cls in seg_classes[cat[i]]:
            pred_set = set(np.where(preds[i, :] == cls)[0])
            label_set = set(np.where(label[i, :] == cls)[0])
            union = len(pred_set.union(label_set))
            inter = len(pred_set.intersection(label_set))
            if union == 0:
                shape_iou += 1
            else:
                shape_iou += inter / union
        shape_iou /= n
        cat_miou[cat[i]][0] += shape_iou
        cat_miou[cat[i]][1] += 1

    return cat_miou


def evaluate(net, test_loader, dev, per_cat_verbose=False):
    category_list = sorted(list(shapenet.seg_classes.keys()))
    eye_mat = np.eye(16)
    net.eval()

    cat_miou = {}
    for k in shapenet.seg_classes.keys():
        cat_miou[k] = [0, 0]
    miou = 0
    count = 0
    per_cat_miou = 0
    per_cat_count = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label, cat in tq:
                num_examples = data.shape[0]
                data = data.to(dev, dtype=torch.float)
                label = label.to(dev, dtype=torch.long)
                cat_ind = [category_list.index(c) for c in cat]
                cat_tensor = (
                    torch.tensor(eye_mat[cat_ind])
                    .to(dev, dtype=torch.float)
                    .repeat(1, 2048)
                )
                cat_tensor = cat_tensor.view(num_examples, -1, 16)
                logits = net(data, cat_tensor).permute(0, 2, 1)
                _, preds = logits.max(1)

                cat_miou = mIoU(
                    preds.cpu().numpy(),
                    label.view(num_examples, -1).cpu().numpy(),
                    cat,
                    cat_miou,
                    shapenet.seg_classes,
                )
                for _, v in cat_miou.items():
                    if v[1] > 0:
                        miou += v[0]
                        count += v[1]
                        per_cat_miou += v[0] / v[1]
                        per_cat_count += 1
                tq.set_postfix(
                    {
                        "mIoU": "%.5f" % (miou / count),
                        "per Category mIoU": "%.5f"
                        % (per_cat_miou / per_cat_count),
                    }
                )
    print(
        "[Test] mIoU: %.5f, per Category mIoU: %.5f"
        % (miou / count, per_cat_miou / per_cat_count)
    )
    if per_cat_verbose:
        print("-" * 60)
        print("Per-Category mIoU:")
        for k, v in cat_miou.items():
            if v[1] > 0:
                print("%s mIoU=%.5f" % (k, v[0] / v[1]))
            else:
                print("%s mIoU=%.5f" % (k, 1))
        print("-" * 60)
    return miou / count, per_cat_miou / per_cat_count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PointTransformerSeg(50, batch_size)

net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))

if args.opt == "sgd":
    # The optimizer strategy described in paper:
    opt = torch.optim.SGD(
        net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[120, 160], gamma=0.1
    )
elif args.opt == "adam":
    # The optimizer strategy proposed by
    # https://github.com/qq456cvb/Point-Transformers:
    opt = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.3)

L = PartSegLoss()

shapenet = ShapeNet(2048, normal_channel=False)

train_loader = CustomDataLoader(shapenet.trainval())
test_loader = CustomDataLoader(shapenet.test())

# Tensorboard
if args.tensorboard:
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import datasets, transforms

    writer = SummaryWriter()
# Select 50 distinct colors for different parts
color_map = torch.tensor(
    [
        [47, 79, 79],
        [139, 69, 19],
        [112, 128, 144],
        [85, 107, 47],
        [139, 0, 0],
        [128, 128, 0],
        [72, 61, 139],
        [0, 128, 0],
        [188, 143, 143],
        [60, 179, 113],
        [205, 133, 63],
        [0, 139, 139],
        [70, 130, 180],
        [205, 92, 92],
        [154, 205, 50],
        [0, 0, 139],
        [50, 205, 50],
        [250, 250, 250],
        [218, 165, 32],
        [139, 0, 139],
        [10, 10, 10],
        [176, 48, 96],
        [72, 209, 204],
        [153, 50, 204],
        [255, 69, 0],
        [255, 145, 0],
        [0, 0, 205],
        [255, 255, 0],
        [0, 255, 0],
        [233, 150, 122],
        [220, 20, 60],
        [0, 191, 255],
        [160, 32, 240],
        [192, 192, 192],
        [173, 255, 47],
        [218, 112, 214],
        [216, 191, 216],
        [255, 127, 80],
        [255, 0, 255],
        [100, 149, 237],
        [128, 128, 128],
        [221, 160, 221],
        [144, 238, 144],
        [123, 104, 238],
        [255, 160, 122],
        [175, 238, 238],
        [238, 130, 238],
        [127, 255, 212],
        [255, 218, 185],
        [255, 105, 180],
    ]
)
# paint each point according to its pred


def paint(batched_points):
    B, N = batched_points.shape
    colored = color_map[batched_points].squeeze(2)
    return colored


best_test_miou = 0
best_test_per_cat_miou = 0

for epoch in range(args.num_epochs):
    print("Epoch #{}: ".format(epoch))
    data, preds, AvgLoss, AvgAcc, training_time = train(
        net, opt, scheduler, train_loader, dev
    )
    if (epoch + 1) % 5 == 0 or epoch == 0:
        test_miou, test_per_cat_miou = evaluate(net, test_loader, dev, True)
        if test_miou > best_test_miou:
            best_test_miou = test_miou
            best_test_per_cat_miou = test_per_cat_miou
            if args.save_model_path:
                torch.save(net.state_dict(), args.save_model_path)
        print(
            "Current test mIoU: %.5f (best: %.5f), per-Category mIoU: %.5f (best: %.5f)"
            % (
                test_miou,
                best_test_miou,
                test_per_cat_miou,
                best_test_per_cat_miou,
            )
        )
    # Tensorboard
    if args.tensorboard:
        colored = paint(preds)
        writer.add_mesh(
            "data", vertices=data, colors=colored, global_step=epoch
        )
        writer.add_scalar(
            "training time for one epoch", training_time, global_step=epoch
        )
        writer.add_scalar("AvgLoss", AvgLoss, global_step=epoch)
        writer.add_scalar("AvgAcc", AvgAcc, global_step=epoch)
        if (epoch + 1) % 5 == 0:
            writer.add_scalar("test mIoU", test_miou, global_step=epoch)
            writer.add_scalar(
                "best test mIoU", best_test_miou, global_step=epoch
            )
    print()
