import argparse
import os
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboardX
import lib
from lib.utils import adjust_learning_rate, cross_entropy_with_label_smoothing, \
    accuracy, save_model, load_model, resume_model


best_val_acc = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image classification')
    parser.add_argument('--dataset', default='imagenet1k',
                        help='Dataset names.')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='The number of classes in the dataset.')
    parser.add_argument('--train_dirs', default='./data/imagenet/train',
                        help='path to training data')
    parser.add_argument('--val_dirs', default='./data/imagenet/val',
                        help='path to validation data')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for val')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument("--color_jitter", action='store_true', default=False,
                        help="To apply color augmentation or not.")
    parser.add_argument('--model', default='axial50s',
                        help='Model names.')
    parser.add_argument('--epochs', type=int, default=130,
                        help='number of epochs to train')
    parser.add_argument('--test_epochs', type=int, default=1,
                        help='number of internal epochs to test')
    parser.add_argument('--save_epochs', type=int, default=1,
                        help='number of internal epochs to save')
    parser.add_argument('--optim', default='sgd',
                        help='Model names.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--warmup_epochs', type=float, default=10,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.00008,
                        help='weight decay')
    parser.add_argument("--label_smoothing", action='store_true', default=False,
                        help="To use label smoothing or not.")
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='To use nesterov or not.')
    parser.add_argument('--work_dirs', default='./work_dirs',
                        help='path to work dirs')
    parser.add_argument('--name', default='axial50s',
                        help='the name of work_dir')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr_scheduler', type=str, default="cosine", choices=["linear", "cosine"],
                        help='how to schedule learning rate')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test')
    parser.add_argument('--test_model', type=int, default=-1,
                        help="Test model's epochs")
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    if not os.path.exists(args.work_dirs):
        os.system('mkdir -p {}'.format(args.work_dirs))
    args.log_dir = os.path.join(args.work_dirs, 'log')
    if not os.path.exists(args.log_dir):
        os.system('mkdir -p {}'.format(args.log_dir))
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.work_dirs = os.path.join(args.work_dirs, args.name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    return args


def val(model, val_loader, criterion, epoch, args, log_writer=False):
    global best_val_acc
    model.eval()
    val_loss = lib.Metric('val_loss')
    val_accuracy = lib.Metric('val_accuracy')

    if epoch == -1:
        epoch = args.epochs - 1

    with tqdm(total=len(val_loader),
              desc='Validate Epoch #{}'.format(epoch + 1)) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(criterion(output, target))
                val_accuracy.update(accuracy(output, target))
                t.update(1)

    print("\nloss: {}, accuracy: {:.2f}, best acc: {:.2f}\n".format(val_loss.avg.item(), 100. * val_accuracy.avg.item(),
                                                                    100. * max(best_val_acc, val_accuracy.avg)))

    if val_accuracy.avg > best_val_acc and log_writer:
        save_model(model, None, -1, args)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
        best_val_acc = max(best_val_acc, val_accuracy.avg)
        log_writer.add_scalar('val/best_acc', best_val_acc, epoch)


def train(model, train_loader, optimizer, criterion, epoch, log_writer, args):
    train_loss = lib.Metric('train_loss')
    train_accuracy = lib.Metric('train_accuracy')
    model.train()
    N = len(train_loader)
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        lr_cur = adjust_learning_rate(args, optimizer, epoch, batch_idx, N, type=args.lr_scheduler)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss.update(loss)
        train_accuracy.update(accuracy(output, target))
        if (batch_idx + 1) % 20 == 0:
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            used_time = time.time() - start_time
            eta = used_time / (batch_idx + 1) * (N - batch_idx)
            eta = str(datetime.timedelta(seconds=int(eta)))
            training_state = '  '.join(['Epoch: {}', '[{} / {}]', 'eta: {}', 'lr: {:.9f}', 'max_mem: {:.0f}',
                                        'loss: {:.3f}', 'accuracy: {:.3f}'])
            training_state = training_state.format(epoch + 1, batch_idx + 1, N, eta, lr_cur, memory,
                                                   train_loss.avg.item(), 100. * train_accuracy.avg.item())
            print(training_state)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def test_net(args):
    print("Init...")
    _, _, val_loader, _ = lib.build_dataloader(args)
    model = lib.build_model(args)
    load_model(model, args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()
    if args.label_smoothing:
        criterion = cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    print("Start testing...")
    val(model, val_loader, criterion, args.test_model, args)


def train_net(args):
    print("Init...")
    log_writer = tensorboardX.SummaryWriter(args.log_dir)
    train_loader, _, val_loader, _ = lib.build_dataloader(args)
    model = lib.build_model(args)
    model = torch.nn.DataParallel(model)
    optimizer = lib.build_optimizer(args, model)

    epoch = 0
    if args.resume:
        epoch = resume_model(model, optimizer, args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    cudnn.benchmark = True

    if args.label_smoothing:
        criterion = cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
    
    print("Start training...")
    while epoch < args.epochs:
        train(model, train_loader, optimizer, criterion, epoch, log_writer, args)

        if (epoch + 1) % args.test_epochs == 0:
            val(model, val_loader, criterion, epoch, args, log_writer)

        if (epoch + 1) % args.save_epochs == 0:
            save_model(model, optimizer, epoch, args)

        epoch += 1

    save_model(model, optimizer, epoch - 1, args)


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test_net(args)
    else:
        train_net(args)


