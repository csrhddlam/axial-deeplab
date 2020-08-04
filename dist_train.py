import argparse
import os
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import tensorboardX
import lib
from lib.utils import adjust_learning_rate, cross_entropy_with_label_smoothing, \
    accuracy, dist_save_model, load_model, resume_model


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
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    args = parser.parse_args()
    if not os.path.exists(args.work_dirs):
        os.system('mkdir -p {}'.format(args.work_dirs))
    args.log_dir = os.path.join(args.work_dirs, 'log')
    if not os.path.exists(args.log_dir):
        os.system('mkdir -p {}'.format(args.log_dir))
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.work_dirs = os.path.join(args.work_dirs, args.name)
    return args


def val(model, val_loader, val_sampler, criterion, epoch, args, log_writer=False, verbose=False):
    global best_val_acc
    model.eval()
    val_loss = lib.Metric('val_loss')
    val_accuracy = lib.Metric('val_accuracy')

    if epoch == -1:
        epoch = args.epochs

    if args.distributed:
        val_sampler.set_epoch(epoch)

    with tqdm(total=len(val_loader),
              desc='Validate Epoch #{}'.format(epoch + 1)) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
                output = model(data)

                loss = criterion(output, target)
                dist.all_reduce(loss)
                pred = output.max(1, keepdim=True)[1]
                acc = pred.eq(target.view_as(pred)).float().mean()
                dist.all_reduce(acc)
               
                val_loss.update(loss * 1.0 / args.ngpus_per_node)
                val_accuracy.update(acc * 1.0 / args.ngpus_per_node)
                t.update(1)

    if verbose:
        print("\nloss: {}, accuracy: {:.2f}, best acc: {:.2f}\n".format(val_loss.avg.item(), 100. * val_accuracy.avg.item(),
                                                                        100. * max(best_val_acc, val_accuracy.avg)))

    if val_accuracy.avg > best_val_acc and log_writer:
        dist_save_model(model, None, -1, args.ngpus_per_node, args)

    if verbose:
        if log_writer:
            log_writer.add_scalar('val/loss', val_loss.avg, epoch)
            log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
            best_val_acc = max(best_val_acc, val_accuracy.avg)
            log_writer.add_scalar('val/best_acc', best_val_acc, epoch)


def train(model, train_sampler, train_loader, optimizer, criterion, epoch, log_writer, args, verbose):
    train_loss = lib.Metric('train_loss')
    train_accuracy = lib.Metric('train_accuracy')
    model.train()
    if args.distributed:
        train_sampler.set_epoch(epoch)
    N = len(train_loader)
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        lr_cur = adjust_learning_rate(args, optimizer, epoch, batch_idx, N, type=args.lr_scheduler)
        if args.cuda:
            data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        dist.all_reduce(loss)
        pred = output.max(1, keepdim=True)[1]
        acc = pred.eq(target.view_as(pred)).float().mean()
        dist.all_reduce(acc)
        train_loss.update(loss * 1.0 / args.ngpus_per_node)
        train_accuracy.update(acc.cpu() * 1.0 / args.ngpus_per_node)
        if (batch_idx + 1) % 20 == 0 and verbose:
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            used_time = time.time() - start_time
            eta = used_time / (batch_idx + 1) * (N - batch_idx)
            eta = str(datetime.timedelta(seconds=int(eta)))
            training_state = '  '.join(['Epoch: {}', '[{} / {}]', 'eta: {}', 'lr: {:.9f}', 'max_mem: {:.0f}',
                                        'loss: {:.3f}', 'accuracy: {:.3f}'])
            training_state = training_state.format(epoch + 1, batch_idx + 1, N, eta, lr_cur, memory,
                                                   train_loss.avg.item(), 100. * train_accuracy.avg.item())
            print(training_state)
    
    if log_writer and verbose:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def train_net(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    verbose = args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    if verbose and not args.test: 
        log_writer = tensorboardX.SummaryWriter(args.log_dir)
    else:
        log_writer = None    
    model = lib.build_model(args)
    optimizer = lib.build_optimizer(args, model)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.val_batch_size = int(args.val_batch_size / ngpus_per_node)        
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    epoch = 0
    if args.resume:
        epoch = resume_model(model, optimizer, args)

    if args.label_smoothing:
        criterion = cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    train_loader, train_sampler, val_loader, val_sampler = lib.build_dataloader(args, args.distributed)

    if args.test:
        load_model(model, args)
        val(model, val_loader, val_sampler, criterion, epoch, verbose=verbose, args=args)
        return
    if verbose:
        print("Start training...")
    while epoch < args.epochs:
        train(model, train_sampler, train_loader, optimizer, criterion, epoch, log_writer, args, verbose=verbose)

        if (epoch + 1) % args.save_epochs == 0:
            dist_save_model(model, optimizer, epoch, ngpus_per_node, args)

        if (epoch + 1) % args.test_epochs == 0:
            val(model, val_loader, val_sampler, criterion, epoch, args, log_writer, verbose=verbose)

        epoch += 1

    dist_save_model(model, optimizer, epoch - 1, ngpus_per_node, args)


def main(args):
    print("Init...")
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main_worker = train_net

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
