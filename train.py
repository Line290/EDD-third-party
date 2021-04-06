import time
import os
import sys
import random
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import EDD
from dataset import *
from utils import *
from metric.metric_score import TEDS
import numpy as np
import pandas as pd

import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from distributed import allreduce_params_opt
from collections import OrderedDict
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # Datasets
    parser.add_argument("--data_folder", type=str, default='output_w_none_399k',
                        help="Directory for dataset.")
    parser.add_argument("--max_len_token_structure", type=int, default=300,
                        help="Don't sample captions structure longer than this length.")
    parser.add_argument("--max_len_token_cell", type=int, default=100,
                        help="Don't sample captions cell longer than this length.")
    parser.add_argument("--image_size", type=int, default=448,
                        help="Different image's height and width for different backbone.")
    # Training
    parser.add_argument("--num_epochs", type=int, default=13,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size for one process.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number workers per process(GPU) to loading data.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--structure_dropout", type=float, default=0.5,
                        help="Dropout ratio of structure module.")
    parser.add_argument("--cell_dropout", type=float, default=0.2,
                        help="Dropout ratio of cell module.")
    parser.add_argument("--grad_clip", type=float,
                        help="Clip gradients at an absolute value of.")
    parser.add_argument("--alpha_c", type=float, default=1.0,
                        help="Regularization parameter for 'doubly stochastic attention', as in the paper.")
    parser.add_argument("--random_seed", type=int, default=123,
                        help="Random seed.")
    parser.add_argument("--print_freq", type=int, default=20,
                        help="Print training/validation stats every __ batches.")
    parser.add_argument("--model_dir", type=str, default='checkpoints',
                        help="Directory for saving models.")
    parser.add_argument("--model_filename", type=str,
                        default='dist',
                        help="Model filename.")
    parser.add_argument("--stage", type=str, default='structure',
                        help="Choice in 'structure' and 'cell' ")
    parser.add_argument("--hyper_loss", type=float, default=1.0,
                        help="when stage is structure, hyper_loss is 1.0. "
                             "When stage is cell, hyper_loss is 0.5, as in the paper.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from saved checkpoint.")
    parser.add_argument("--pretrained_model_path", type=str,
                        default=None,
                        help="Resume training from saved checkpoint.")
    parser.add_argument("--first_epoch", type=int, default=10,
                        help="Number of epoch in learning rate 1e-3.")
    parser.add_argument("--second_epoch", type=int, default=3,
                        help="Number of epoch in learning rate 1e-4.")
    parser.add_argument("--p_structure", type=float, default=1.0,
                        help="probability of using gt token to predict next token.")
    parser.add_argument("--p_cell", type=float, default=1.0,
                        help="probability of using gt token to predict next token.")

    # Validation
    parser.add_argument("--only_val", action="store_true",
                        help="Only validation.")

    # Model setting
    parser.add_argument("--backbone", type=str, default='resnet18',
                        help="The backbone of encoder")
    parser.add_argument("--EDD_type", type=str, default='S1S1',
                        help="The type of EDD, choice in S1S1, S2S2")
    parser.add_argument("--emb_dim_structure", type=int, default=16,
                        help="Dimension of word embeddings for structure token")
    parser.add_argument("--emb_dim_cell", type=int, default=80,
                        help="Dimension of word embeddings for cell token")
    parser.add_argument("--attention_dim", type=int, default=512,
                        help="Dimension of attention linear layers")
    parser.add_argument("--decoder_dim_structure", type=int, default=256,
                        help="Dimension of decoder RNN structure")
    parser.add_argument("--decoder_dim_cell", type=int, default=512,
                        help="Dimension of decoder RNN cell")
    parser.add_argument("--fp16", action="store_true",
                        help="Model with FP16.")
    args = parser.parse_args()


    # Create structure table and cell table
    word_map_structure_file = os.path.join(
        args.data_folder, "WORDMAP_STRUCTURE.json")
    word_map_cell_file = os.path.join(args.data_folder, "WORDMAP_CELL.json")
    with open(word_map_structure_file, "r") as j:
        word_map_structure = json.load(j)
    with open(word_map_cell_file, "r") as j:
        word_map_cell = json.load(j)
    id2word_structure = id_to_word(word_map_structure)
    id2word_cell = id_to_word(word_map_cell)

    # Init. distribution training setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
    random_seed = args.random_seed + args.local_rank
    set_random_seeds(random_seed=random_seed)
    torch.cuda.set_device(args.local_rank % torch.cuda.device_count())  # each node has same number of GPUs
    print('| distributed init (rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.local_rank)
    torch.distributed.barrier()

    # Create metrics
    teds = TEDS(n_jobs=64//torch.cuda.device_count())

    # Create save path
    save_folder = os.path.join(args.model_dir, args.backbone+'_'+args.EDD_type)
    args.save_folder = save_folder
    if args.local_rank == 0:
        if os.path.exists(save_folder) is False:
            os.makedirs(save_folder)
    torch.distributed.barrier()

    # Device ID in each node
    device_id = torch.cuda.current_device()
    device = torch.device("cuda:%s"%device_id)

    # Init. training
    start_epoch = 0
    # keeps track of number of epochs since there's been an improvement in validation
    epochs_since_improvement = 0
    best_TED = 0.  # TED score right now
    logger_file = os.path.join(save_folder, 'logger.txt')

    # structure_weight = pd.read_csv('structure_class_weight.csv').values[:, -1].astype(np.float32)
    # structure_weight[structure_weight>1.] = structure_weight[structure_weight>1.]*2
    # print(structure_weight)
    # criterion_structure = nn.CrossEntropyLoss(reduction='mean', weight=torch.FloatTensor(structure_weight))
    criterion_structure = nn.CrossEntropyLoss(reduction='mean')
    criterion_cell = nn.CrossEntropyLoss(reduction='mean')

    if args.EDD_type == 'S1S1':
        encoded_image_size = args.image_size // 16
        last_conv_stride = 1
    elif args.EDD_type == 'S2S2':
        encoded_image_size = args.image_size // 32
        last_conv_stride = 2

    model = EDD(encoded_image_size=encoded_image_size, # feature's size of last Conv.
                encoder_dim=512,  # feature's channel of last Conv.
                pretrained=False,  # pretrained backbone network
                structure_attention_dim=args.attention_dim,
                structure_embed_dim=args.emb_dim_structure,
                structure_decoder_dim=args.decoder_dim_structure,
                structure_dropout=args.structure_dropout,
                structure_vocab=word_map_structure,
                cell_attention_dim=args.attention_dim,
                cell_embed_dim=args.emb_dim_cell,
                cell_decoder_dim=args.decoder_dim_cell,
                cell_dropout=args.cell_dropout,
                cell_vocab=word_map_cell,
                criterion_structure=criterion_structure,
                criterion_cell=criterion_cell,
                alpha_c=args.alpha_c,
                id2word_structure=id2word_structure,
                id2word_cell=id2word_cell,
                last_conv_stride=last_conv_stride,
                lstm_bias=True,  # https://github.com/pytorch/pytorch/issues/42605
                backbone=args.backbone)

    # Move to GPU, if available
    model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    # https://github.com/pytorch/pytorch/issues/24005
    model = DDP(model, delay_allreduce=True)
    model._disable_allreduce = True
    if is_main_process():
        count = 0
        model_name_list = []
        for name, param in model.named_parameters():
            print(name, param.requires_grad, param.data.size())
            count += np.prod(np.array(param.data.size()))
            model_name_list.append(name)
        print(count)

    if args.pretrained_model_path is not None:
        checkpoint = torch.load(args.pretrained_model_path,
                                map_location='cpu')
        model.module.load_state_dict(checkpoint["model"], strict=False)
        if is_main_process():
            print("Load pretrained model from: ", args.pretrained_model_path)

        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            best_TED = checkpoint['ted_score']
            start_epoch = checkpoint['epoch'] + 1
            if is_main_process():
                print("Start epoch: %d, best TED score: %.4f"%(start_epoch, best_TED), flush=True)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Local Rank: {}, Loading train_loader and val_loader:".format(args.local_rank))

    val_set = CaptionDataset(
        args.data_folder,
        'val',
        transform=transforms.Compose([normalize]),
        max_len_token_structure=args.max_len_token_structure,
        max_len_token_cell=args.max_len_token_cell,
        image_size=args.image_size)
    val_sampler = DistributedSampler(dataset=val_set)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    if args.only_val:
        print("Local Rank: {}, Only Validation ...".format(args.local_rank))
        recent_ted_score = val(val_loader=val_loader,
                               model=model,
                               device=device,
                               args=args,
                               teds=teds,
                               logger_file=logger_file)
        return recent_ted_score

    train_set = CaptionDataset(
        args.data_folder,
        ['train'],
        transform=transforms.Compose([normalize]),
        max_len_token_structure=args.max_len_token_structure,
        max_len_token_cell=args.max_len_token_cell,
        image_size=args.image_size)
    train_sampler = DistributedSampler(dataset=train_set)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True)
    print("Local Rank: {}, Done train_loader and val_loader:".format(args.local_rank))


    # Train for each epoch
    for epoch in range(start_epoch, args.num_epochs):
        # Structure
        if args.first_epoch <= epoch < args.first_epoch + args.second_epoch:
            adjust_learning_rate(optimizer, 0.1*args.learning_rate)
        elif args.first_epoch + args.second_epoch <= epoch < args.first_epoch*4 + args.second_epoch: # cell
            args.stage = 'cell'
            args.hyper_loss = 0.5
            best_TED = 0.
            adjust_learning_rate(optimizer, args.learning_rate)
        elif args.first_epoch*4 + args.second_epoch <= epoch < args.first_epoch*6 + args.second_epoch: # cell
            best_TED = 0.
            adjust_learning_rate(optimizer, 0.5*args.learning_rate)
        elif args.first_epoch * 6 + args.second_epoch <= epoch < args.first_epoch * 8 + args.second_epoch:  # cell
            best_TED = 0.
            adjust_learning_rate(optimizer, 0.1 * args.learning_rate)
        elif args.first_epoch * 8 + args.second_epoch <= epoch < args.first_epoch * 9 + args.second_epoch:  # cell
            best_TED = 0.
            adjust_learning_rate(optimizer, 0.05*args.learning_rate)
        elif epoch >= args.first_epoch*9 + args.second_epoch:
            adjust_learning_rate(optimizer, 0.01*args.learning_rate)
        else:
            pass

        if is_main_process():
            print("Epoch: {}, Stage: {}, hyper_loss: {}, lr: {} ...".format(
                epoch,
                args.stage,
                args.hyper_loss,
                optimizer.state_dict()['param_groups'][0]['lr']))

        print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))
        train(train_loader=train_loader,
              model=model,
              optimizer=optimizer,
              epoch=epoch,
              device=device,
              args=args,
              logger_file=logger_file)

        print("Local Rank: {}, Epoch: {}, Validation ...".format(args.local_rank, epoch))
        recent_ted_score = val(val_loader=val_loader,
                               model=model,
                               device=device,
                               args=args,
                               teds=teds,
                               logger_file=logger_file)

        # Check if there was an improvement
        is_best = recent_ted_score > best_TED
        best_TED = max(recent_ted_score, best_TED)
        if not is_best:
            epochs_since_improvement += 1
            if is_main_process():
                print("\nEpochs since last improvement: %d\n" %
                      (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # save checkpoint
        filename = os.path.join(save_folder,
                                args.model_filename + '_' +
                                args.stage+ '_epoch_'+ str(epoch)
                                + "_score_"+str(recent_ted_score)[:6] + '.pth.tar')
        save_on_master(epoch, epochs_since_improvement, model,
                       optimizer, recent_ted_score, is_best, filename)


def train(train_loader, model, optimizer, epoch, device, args, logger_file):

    model.train()
    # model.module.encoder_wrapper.eval()
    world_size = args.world_size

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    losses_structure = AverageMeter()
    losses_cell = AverageMeter()

    top5accs_structure = AverageMeter()  # top5 accuracy
    top1accs_structure = AverageMeter()  # top1 accuracy
    top5accs_cell = AverageMeter()  # top5 accuracy
    top1accs_cell = AverageMeter()  # top1 accuracy

    start = time.time()
    if is_main_process():
        print("length of train_loader: {}".format(len(train_loader)))
    for i, (imgs,
            caption_structures,
            caplen_structures, 
            caption_cells, 
            caplen_cells, 
            number_cell_per_images) in enumerate(train_loader):
        if epoch == 0:
            adjust_learning_rate(optimizer, min((i/500.), 1.)*args.learning_rate)
        
        data_time.update(time.time() - start)
        
        imgs = imgs.to(device)
        caption_structures = caption_structures.to(device)
        caplen_structures = caplen_structures.to(device)
        caption_cells = [caption_cell.to(device) for caption_cell in caption_cells]
        caplen_cells = [caplen_cell.to(device) for caplen_cell in caplen_cells]
        number_cell_per_images = number_cell_per_images.to(device)
        
        loss_structures, scores_structure, targets_structure, \
        loss_cells, total_scores_cells, total_target_cells = \
            model(imgs,
                  caption_structures,
                  caplen_structures,
                  caption_cells,
                  caplen_cells,
                  number_cell_per_images)

        # Total loss
        if args.stage == 'cell':
            loss = args.hyper_loss * loss_structures + (1-args.hyper_loss) * loss_cells
        elif args.stage == 'structure':
            loss = args.hyper_loss * loss_structures + (1 - args.hyper_loss) * loss_cells.detach()
        # if args.fp16:
        #     optimizer.zero_grad()
        #     scaler.scale(loss).backward()   # gradients are scaled
        #     if args.grad_clip is not None:
        #         # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        #         scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        #         # clip_gradient(optimizer, args.grad_clip)
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        
        # Back prop.
        iters_to_accumulate = 1
        with amp.scale_loss(loss/iters_to_accumulate, optimizer) as scaled_loss:
            scaled_loss.backward()
        if i% iters_to_accumulate == 0:
            allreduce_params_opt(optimizer)
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), args.grad_clip) # actually optimizer
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            # Keep track of metrics
            # all reduce LOSS
            torch.distributed.all_reduce(loss_structures)
            torch.distributed.all_reduce(loss_cells)
            torch.distributed.all_reduce(loss)

            losses_structure.update(loss_structures.item()/world_size, 1)
            losses_cell.update(loss_cells.item() / world_size, 1)
            losses.update(loss.item() / world_size, 1)

            # STRUCTURE ACC.
            top1_structure, top5_structure = accuracy(scores_structure, targets_structure, (1, 5))
            targets_structure_size = torch.LongTensor([targets_structure.size(0)]).squeeze(0).to(device)
            torch.distributed.all_reduce(top1_structure)
            torch.distributed.all_reduce(top5_structure)
            torch.distributed.all_reduce(targets_structure_size)
            top5accs_structure.update(top5_structure.item()/world_size, targets_structure_size.item())
            top1accs_structure.update(top1_structure.item()/world_size, targets_structure_size.item())

            # CELL ACC.
            top1_cell, top5_cell = accuracy(total_scores_cells, total_target_cells, (1, 5))
            total_target_cells_size = torch.LongTensor([total_target_cells.size(0)]).squeeze(0).to(device)
            torch.distributed.all_reduce(top1_cell)
            torch.distributed.all_reduce(top5_cell)
            torch.distributed.all_reduce(total_target_cells_size)
            top5accs_cell.update(top5_cell.item()/world_size, total_target_cells_size.item())
            top1accs_cell.update(top1_cell.item()/world_size, total_target_cells_size.item())
            if is_main_process():
                print('Epoch: [{0}][{1}/{2}]\t'
                      'lr: {lr:.8f}'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss Stru {loss_s.val:.4f} ({loss_s.avg:.4f})\t'
                      'Loss Cell {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
                      'Top-5 Stru. Acc. {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Top-1 Stru. Acc. {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top-5 Cell Acc. {top5_s.val:.3f} ({top5_s.avg:.3f})\t'
                      'Top-1 Cell Acc. {top1_s.val:.3f} ({top1_s.avg:.3f})'.format(
                        epoch, i, len(train_loader),
                        lr=optimizer.param_groups[0]['lr'],
                        batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        loss_s=losses_structure, loss_c=losses_cell,
                        top5=top5accs_structure, top1=top1accs_structure,
                        top5_s=top5accs_cell, top1_s=top1accs_cell), flush=True)

        # if i % 10000 == 0:
        #     filename = os.path.join(args.save_folder,
        #                             args.model_filename + '_' +
        #                             args.stage + '_epoch_' + str(epoch)
        #                             + "_iters_" + str(i) + '.pth.tar')
        #     save_on_master(epoch, 1, model,
        #                    optimizer, 0.0, False, filename)

    if is_main_process():
        log_str = ('Epoch: [{0}]\n'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Loss Stru {loss_s.val:.4f} ({loss_s.avg:.4f})\t'
                   'Loss Cell {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
                   'Top-5 Stru. Acc. {top5.val:.3f} ({top5.avg:.3f})\t'
                   'Top-1 Stru. Acc. {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Top-5 Cell Acc. {top5_s.val:.3f} ({top5_s.avg:.3f})\t'
                   'Top-1 Cell Acc. {top1_s.val:.3f} ({top1_s.avg:.3f})'.format(
                   epoch,
                   loss=losses,
                   loss_s=losses_structure, loss_c=losses_cell,
                   top5=top5accs_structure, top1=top1accs_structure,
                   top5_s=top5accs_cell, top1_s=top1accs_cell))
        with open(logger_file, 'a') as f:
            f.write(log_str+'\n')



def val(val_loader, model, device, args, teds, logger_file):
    model.eval()
    world_size = args.world_size

    total_loss_structure = list()
    total_loss_cell = list()
    total_loss = list()

    top5accs_structure = AverageMeter()  # top5 accuracy
    top1accs_structure = AverageMeter()  # top1 accuracy
    top5accs_cell = AverageMeter()  # top5 accuracy
    top1accs_cell = AverageMeter()  # top1 accuracy

    html_predict_only_structures = list()
    html_true_only_structures = list()
    html_predict_only_cells = list()
    html_predict_alls = list()
    html_trues = list()

    with torch.no_grad():
        for it, (imgs, caption_structures, caplen_structures, caption_cells, caplen_cells, number_cell_per_images) in enumerate(val_loader):
            imgs = imgs.to(device)
            caption_structures = caption_structures.to(device)
            caplen_structures = caplen_structures.to(device)
            caption_cells = [caption_cell.to(device) for caption_cell in caption_cells]
            caplen_cells = [caplen_cell.to(device) for caplen_cell in caplen_cells]

            loss_structures, loss_cells, \
            batch_html_predict_only_structures, \
            batch_html_true_only_structures, \
            batch_html_predict_only_cells, \
            batch_html_predict_alls, \
            batch_html_trues, \
            scores_structure, targets_structure, \
            total_scores_cells, total_target_cells = model(imgs,
                               caption_structures,
                               caplen_structures,
                               caption_cells,
                               caplen_cells,
                               number_cell_per_images)

            loss = args.hyper_loss * loss_structures + \
                   (1-args.hyper_loss) * loss_cells

            total_loss_structure.append(loss_structures.cpu())
            total_loss_cell.append(loss_cells.cpu())
            total_loss.append(loss.cpu())

            html_predict_only_structures.extend(batch_html_predict_only_structures)
            html_true_only_structures.extend(batch_html_true_only_structures)
            html_predict_only_cells.extend(batch_html_predict_only_cells)
            html_predict_alls.extend(batch_html_predict_alls)
            html_trues.extend(batch_html_trues)

            top1_structure, top5_structure = accuracy(scores_structure, targets_structure, (1, 5))
            targets_structure_size = torch.LongTensor([targets_structure.size(0)]).squeeze(0).to(device)
            torch.distributed.all_reduce(top1_structure)
            torch.distributed.all_reduce(top5_structure)
            torch.distributed.all_reduce(targets_structure_size)
            top5accs_structure.update(top5_structure.item() / world_size, targets_structure_size.item())
            top1accs_structure.update(top1_structure.item() / world_size, targets_structure_size.item())

            # CELL ACC.
            top1_cell, top5_cell = accuracy(total_scores_cells, total_target_cells, (1, 5))
            total_target_cells_size = torch.LongTensor([total_target_cells.size(0)]).squeeze(0).to(device)
            torch.distributed.all_reduce(top1_cell)
            torch.distributed.all_reduce(top5_cell)
            torch.distributed.all_reduce(total_target_cells_size)
            top5accs_cell.update(top5_cell.item() / world_size, total_target_cells_size.item())
            top1accs_cell.update(top1_cell.item() / world_size, total_target_cells_size.item())
            if is_main_process():
                print('it [{0}/{1}]\t'
                      'Top-5 Stru. Acc. {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Top-1 Stru. Acc. {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top-5 Cell Acc. {top5_s.val:.3f} ({top5_s.avg:.3f})\t'
                      'Top-1 Cell Acc. {top1_s.val:.3f} ({top1_s.avg:.3f})'.format(
                    it, len(val_loader),
                    top5=top5accs_structure, top1=top1accs_structure,
                    top5_s=top5accs_cell, top1_s=top1accs_cell), flush=True)

        # Average loss in local rank
        loss_structures = mean_loss(total_loss_structure).to(device)
        loss_cells = mean_loss(total_loss_cell).to(device)
        loss = mean_loss(total_loss).to(device)

        # Average loss in all rank
        torch.distributed.all_reduce(loss_structures)
        torch.distributed.all_reduce(loss_cells)
        torch.distributed.all_reduce(loss)

        # Calculate val. set size
        pred_local_rank_size = torch.LongTensor([len(html_predict_alls)]).squeeze(0).to(device)
        torch.distributed.all_reduce(pred_local_rank_size)

        # Calculate average TEDS scores in local rank
        scores_only_structure = teds.batch_evaluate_html(
            html_predict_only_structures, html_true_only_structures, is_tqdm=is_main_process())

        scores_only_cell = teds.batch_evaluate_html(
            html_predict_only_cells, html_trues, is_tqdm=is_main_process())

        scores_all = teds.batch_evaluate_html(
            html_predict_alls, html_trues, is_tqdm=is_main_process())

        torch.distributed.barrier()
        if is_main_process():
            for ii in range(3):
                print("#" * 80)
                print("index: ", ii)
                print("html_predict_only_structure: \n", html_predict_only_structures[ii])
                print("html_true_only_structure: \n", html_true_only_structures[ii])
                print('TEDS score only structure:', scores_only_structure[ii])
                print("-" * 80)
                print("html_predict_only_cell: \n", html_predict_only_cells[ii])
                print("html_true: \n", html_trues[ii])
                print('TEDS score only cell:', scores_only_cell[ii])
                print("-" * 80)
                print("html_predict_all: \n", html_predict_alls[ii])
                print("html_true: \n", html_trues[ii])
                print('TEDS score:', scores_all[ii])
                sys.stdout.flush()
        torch.distributed.barrier()

        ted_score_structure = torch.FloatTensor([np.mean(scores_only_structure)]).squeeze(0).to(device)
        ted_score_cell = torch.FloatTensor([np.mean(scores_only_cell)]).squeeze(0).to(device)
        ted_score = torch.FloatTensor([np.mean(scores_all)]).squeeze(0).to(device)

        # Calculate average TEDS scores in all rank
        torch.distributed.all_reduce(ted_score_structure)
        torch.distributed.all_reduce(ted_score_cell)
        torch.distributed.all_reduce(ted_score)
        if is_main_process():
            print_lines = [
                "Eval set size: {}".format(pred_local_rank_size.item()),
                "LOSS_STRUCTURE: {} \nLOSS_CELL: {} \nLOSS_DUAL_DECODER: {}".format(
                    loss_structures.item() / world_size, loss_cells.item() / world_size, loss.item() / world_size),
                "TED_SCORE_STRUCTURE: {}".format(ted_score_structure.item() / world_size),
                "TED_SCORE_CELL: {}".format(ted_score_cell.item() / world_size),
                "TED_SCORE: {}".format(ted_score.item() / world_size)
            ]
            with open(logger_file, 'a') as f:
                for line in print_lines:
                    print(line, flush=True)
                    f.write(line + '\n')
                f.write('\n')
        if args.stage == 'structure':
            return ted_score_structure.item() / world_size
        else:
            return ted_score.item()  / world_size


if __name__ == "__main__":
    main()
