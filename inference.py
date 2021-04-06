import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
from utils import *
from models import EDD
from metric.metric_score import TEDS
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encoderImage(encoder, image_path, image_size):
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (image_size, image_size))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  #

    # Encode
    image = image.unsqueeze(0)  # (1, 3, img_size, img_size)
    # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_out_structure, encoder_out_cell = encoder(image)
    return encoder_out_structure, encoder_out_cell


def structure_image_beam_search(encoder_out_structure, decoder, word_map, structure_weight=None,
                                beam_size=3, max_seq_len=300, rank_method='sum', T=0.6):
    is_overflow = False
    k = beam_size
    vocab_size = len(word_map)
    decoder_structure_dim = 256
    # Read image and process
    encoder_dim = encoder_out_structure.size(3)  # 512

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out_structure = encoder_out_structure.view(1, -1, encoder_dim)
    num_pixels = encoder_out_structure.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out_structure = encoder_out_structure.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step decode structure; construct just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # tensor save hidden state and after filter to choice hidden state to pass cell decoder
    seqs_hidden_states = torch.zeros(k, 1, decoder_structure_dim).to(device)

    # Lists to store completed sequences, their alphas and scores, hidden
    complete_seqs = list()
    complete_seqs_scores = list()
    complete_seqs_hiddens = list()

    # start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out_structure)  # h, c: (k, decoder_structure_dim)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)   (s, 16)

        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out_structure, h)

        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        if structure_weight is not None:
            scores = scores * structure_weight
        scores = scores / T
        scores = F.log_softmax(scores, dim=1)

        if step == 1:
            # scores's shape: (1, vocab_size)
            top_k_scores, top_k_words = (top_k_scores+scores)[0].topk(k, 0, True, True)
        else:
            # scores's shape: (s, vocab_size)
            if rank_method == 'mean':
                top_k_scores, top_k_words = ((top_k_scores*(step-1)+scores)/step).view(-1).topk(k, 0, True, True)
            elif rank_method == 'sum':
                top_k_scores, top_k_words = (top_k_scores+scores).view(-1).topk(k, 0, True, True)
            else:
                RuntimeError("Invalid rank method: ", rank_method)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas, and hidden_state
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        if step == 1:
            seqs_hidden_states = h.unsqueeze(1)
        else:
            seqs_hidden_states = torch.cat(
                [seqs_hidden_states[prev_word_inds], h[prev_word_inds].unsqueeze(1)], dim=1)  # (s, step+1, decoder_structure_dim)
            # Which sequences are incomplete (didn't reach <end>)?

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        # Break if things have been going on too long
        if step > max_seq_len:
            incomplete_inds = []
            is_overflow = True

        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_hiddens.extend(
                seqs_hidden_states[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_hidden_states = seqs_hidden_states[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]

        c = c[prev_word_inds[incomplete_inds]]
        encoder_out_structure = encoder_out_structure[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        step += 1

    max_score = max(complete_seqs_scores)
    i = complete_seqs_scores.index(max_score)
    seq = complete_seqs[i]
    hidden_states = complete_seqs_hiddens[i]

    return seq, hidden_states, is_overflow, max_score.cpu().numpy()


def cell_image_beam_search(encoder_out, decoder, word_map, hidden_state_structure,
                           beam_size=3., max_seq_len=100, rank_method='sum', T=0.6):
    is_overflow = False
    k = beam_size
    vocab_size = len(word_map)
    encoder_dim = encoder_out.size(3)
    decoder_structure_dim = 256

    # Flatten encoding
    # (1, num_pixels, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    # (k, num_pixels, encoder_dim)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step decode structure; construct just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    complete_seqs = list()
    complete_seqs_scores = list()
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(
            k_prev_words).squeeze(1)  # (s, embed_dim)

        s = list(encoder_out.size())[0]
        hidden_state_structure_s = hidden_state_structure.expand(
            s, decoder_structure_dim)
        # (s, encoder_dim), (s, num_pixels)
        awe, alpha = decoder.attention(encoder_out, h, hidden_state_structure_s)

        # gating scalar, (s, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = scores / T
        scores = F.log_softmax(scores, dim=1)

        if step == 1:
            # scores's shape: (1, vocab_size)
            top_k_scores, top_k_words = (top_k_scores+scores)[0].topk(k, 0, True, True)
        else:
            # scores's shape: (s, vocab_size)
            if rank_method == 'mean':
                top_k_scores, top_k_words = ((top_k_scores*(step-1)+scores)/step).view(-1).topk(k, 0, True, True)
            elif rank_method == 'sum':
                top_k_scores, top_k_words = (top_k_scores+scores).view(-1).topk(k, 0, True, True)
            else:
                RuntimeError("Invalid rank method: ", rank_method)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas, and hidden_state
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]

        # Break if things have been going on too long
        if step > max_seq_len:
            incomplete_inds = []
            is_overflow = True

        complete_inds = list(
            set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]

        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        step += 1

    max_score = max(complete_seqs_scores)
    i = complete_seqs_scores.index(max_score)
    seq = complete_seqs[i]
    return seq, is_overflow, max_score.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument("--start_idx", type=int, default=0,
                        help="start indices in all test images.")
    parser.add_argument("--offset", type=int, default=1000,
                        help="we test start_idx: start_idx+offset in all test images.")

    parser.add_argument('--img', '-i', default='img.png', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map_structure', '-wms',
                        help='path to word map structure JSON')
    parser.add_argument('--word_map_cell', '-wmc',
                        help='path to word map cell JSON')
    parser.add_argument('--beam_size_structure', '-bs', default=3,
                        type=int, help='beam size for beam search')
    parser.add_argument('--beam_size_cell', '-bc', default=3,
                        type=int, help='beam size for beam search')
    parser.add_argument("--max_seq_len_structure", type=int, default=300,
                        help="Maximal number of tokens generated by structure decoder")
    parser.add_argument("--max_seq_len_cell", type=int, default=100,
                        help="Maximal number of tokens generated by cell decoder")
    parser.add_argument('--dont_smooth', dest='smooth',
                        action='store_false', help='do not smooth alpha overlay')
    parser.add_argument("--img_from_val", action="store_true",
                        help="image from validation set.")
    parser.add_argument("--data_folder", type=str, default='output_w_none_399k_memory_effi',
                        help="Directory for dataset.")
    parser.add_argument("--split", type=str, default='val',
                        help="evaluate part.")
    parser.add_argument("--all", action="store_true",
                        help="All test samples.")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of test samples.")
    parser.add_argument("--print_freq", type=int, default=1000,
                        help="Print result.")
    parser.add_argument("--not_save", action="store_true",
                        help="Save final results")
    parser.add_argument("--T", type=float, default=0.6,
                        help="Temperature.")

    # Model setting
    parser.add_argument("--backbone", type=str, default='resnet18',
                        help="The backbone of encoder")
    parser.add_argument("--EDD_type", type=str, default='S2S2',
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
    parser.add_argument("--image_size", type=int, default=448,
                        help="Different image's height and width for different backbone.")
    parser.add_argument("--rank_method", type=str, default='mean',
                        help="The method of rank beam search, choosing in 'mean' and 'sum'.")

    args = parser.parse_args()
    teds = TEDS(n_jobs=8)
    split = args.split

    # Load image path
    with open(os.path.join(args.data_folder, split + '_IMAGE_PATHS.txt'), 'r') as f:
        img_paths = []
        for line in f:
            img_paths.append(line.strip())
    print("Split: %s, number of images: %d" % (split, len(img_paths)))

    if split == 'val':
        # Load encoded captions structure
        with open(os.path.join(args.data_folder, split + '_CAPTIONS_STRUCTURE' + '.json'), 'r') as j:
            captions_structure = json.load(j)

        # Load caption structure length (completely into memory)
        with open(os.path.join(args.data_folder, split + '_CAPLENS_STRUCTURE' + '.json'), 'r') as j:
            caplens_structure = json.load(j)

        # Load encoded captions cell
        with open(os.path.join(args.data_folder, split + '_CAPTIONS_CELL' + '.json'), 'r') as j:
            captions_cell = json.load(j)
        # Load caption cell length
        with open(os.path.join(args.data_folder, split + '_CAPLENS_CELL' + '.json'), 'r') as j:
            caplens_cell = json.load(j)
        with open(os.path.join(args.data_folder, split + "_NUMBER_CELLS_PER_IMAGE.json"), "r") as j:
            number_cell_per_images = json.load(j)

    with open(args.word_map_structure, 'r') as j:
        word_map_structure = json.load(j)
    with open(args.word_map_cell, "r") as j:
        word_map_cell = json.load(j)
    id2word_stucture = id_to_word(word_map_structure)
    id2word_cell = id_to_word(word_map_cell)

    # Load model
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    if args.EDD_type == 'S1S1':
        encoded_image_size = args.image_size // 16
        last_conv_stride = 1
    elif args.EDD_type == 'S2S2':
        encoded_image_size = args.image_size // 32
        last_conv_stride = 2

    model = EDD(encoded_image_size=encoded_image_size,
                encoder_dim=512,
                pretrained=False,
                structure_attention_dim=args.attention_dim,
                structure_embed_dim=args.emb_dim_structure,
                structure_decoder_dim=args.decoder_dim_structure,
                structure_dropout=0.,
                structure_vocab=word_map_structure,
                cell_attention_dim=args.attention_dim,
                cell_embed_dim=args.emb_dim_cell,
                cell_decoder_dim=args.decoder_dim_cell,
                cell_dropout=0.,
                cell_vocab=word_map_cell,
                criterion_structure=criterion,
                criterion_cell=criterion,
                alpha_c=1.,
                id2word_structure=id2word_stucture,
                id2word_cell=id2word_cell,
                last_conv_stride=last_conv_stride,
                lstm_bias=True,
                backbone=args.backbone)
    model = model.to(device)

    checkpoint = torch.load(args.model, map_location=str(device))

    # try:
    #     model.load_state_dict(checkpoint["model"])
    # except:
    #     reform_checkpoint = OrderedDict()
    #     for k, v in checkpoint["model"].items():
    #         new_k = k
    #         if 'resnet' in k:
    #             new_k = k.replace('resnet', 'backbone_net')
    #         reform_checkpoint[new_k] = v
    #     model.load_state_dict(reform_checkpoint)
    model.load_state_dict(checkpoint["model"], strict=False)
    #structure_weight = pd.read_csv('structure_class_weight.csv').values[:, -1].astype(np.float32)
    #structure_weight = (torch.cuda.FloatTensor(structure_weight)-1)/20.+ 1.
    structure_weight = None
    model.eval()

    encoder = model.encoder
    decoder_structure = model.decoder_structure
    decoder_cell = model.decoder_cell
    encoder.eval()
    decoder_cell.eval()
    decoder_structure.eval()

    pred_html_only_structures = []
    gt_html_only_structures = []
    pred_html_alls = []
    gt_html_alls = []
    skipped_idx = set()
    test_img_paths = []

    max_score_structure_save = []
    max_score_cell_mean_save = []

    if args.all:
        n_samples = len(img_paths)
    else:
        n_samples = args.n_samples
    for img_idx, img_path in tqdm(enumerate(img_paths[args.start_idx:args.start_idx + args.offset])):
        img_index = img_idx + args.start_idx
        args.img = img_paths[img_index]
        test_img_paths.append(img_paths[img_index])
        with torch.no_grad():
            encoder_out_structure, encoder_out_cell = encoderImage(encoder, args.img, args.image_size)

            seq, hidden_states, is_overflow_structure, max_score_structure = structure_image_beam_search(
                encoder_out_structure, decoder_structure, word_map_structure,
                beam_size=args.beam_size_structure,
                max_seq_len=args.max_seq_len_structure,
                rank_method=args.rank_method,
                T=args.T,
                structure_weight=structure_weight)
            if is_overflow_structure:
                print("skip {0}, length of generated structure's token larger than {1}.".format(
                    img_index, args.max_seq_len_structure))
                skipped_idx.add(img_index)

            cells = []
            max_score_cell_list = []
            html = ""
            html_only_structure = ""
            is_overflow_cell = False
            for index, s in enumerate(seq[1:-1]):  # ignore <start> and <end>
                html += id2word_stucture[s]
                html_only_structure += id2word_stucture[s]
                if id2word_stucture[s] == "<td>" or id2word_stucture[s] == ">":
                    hidden_state_structure = hidden_states[index+1]
                    seq_cell, is_overflow_cell, max_score_cell = cell_image_beam_search(
                        encoder_out_cell, decoder_cell, word_map_cell, hidden_state_structure,
                        beam_size=args.beam_size_cell,
                        max_seq_len=args.max_seq_len_cell,
                        rank_method=args.rank_method,
                        T=args.T)
                    max_score_cell_list.append(max_score_cell)
                    if is_overflow_cell:
                        print("skip {0}, length of generated cell's token larger than {1}.".format(
                            img_index, args.max_seq_len_cell))
                        skipped_idx.add(img_index)
                        # break
                    html_cell = convertId2wordSentence(id2word_cell, seq_cell)
                    html += html_cell

            if args.split == 'val':
                one_captions_structure = captions_structure[img_index]
                one_captions_cell = captions_cell[img_index]
                gt_html = ""
                gt_html_only_structure = ""
                cell_index = 0
                for s in one_captions_structure[1:-1]:  # ignore <start> and <end>
                    gt_html += id2word_stucture[s]
                    gt_html_only_structure += id2word_stucture[s]
                    if id2word_stucture[s] == "<td>" or id2word_stucture[s] == ">":
                        seq_cell = one_captions_cell[cell_index]
                        html_cell = convertId2wordSentence(id2word_cell, seq_cell)
                        gt_html += html_cell
                        cell_index += 1
                gt_html_only_structures.append(create_html(gt_html_only_structure))
                gt_html_alls.append(create_html(gt_html))

            pred_html_only_structures.append(create_html(html_only_structure))
            pred_html_alls.append(create_html(html))


            max_score_cell_mean = np.mean(max_score_cell_list)
            max_score_structure_save.append(max_score_structure)
            max_score_cell_mean_save.append(max_score_cell_mean)
            if img_idx % args.print_freq == 0:
                print("Index: ", img_index)
                print("#" * 80)
                print("Pred. only structure: ")
                print(html_only_structure)
                if args.split == 'val':
                    print("GT. only structure: ")
                    print(gt_html_only_structure)
                    print("ted score only structure: ", teds.evaluate(create_html(html_only_structure),
                                                                      create_html(gt_html_only_structure)))
                print("#"*80)
                print("Pred html all: ")
                print(html)
                if args.split == 'val':
                    print("GT html all: ")
                    print(gt_html)
                    print("ted score all: ", teds.evaluate(create_html(html),
                                                           create_html(gt_html)))
                sys.stdout.flush()
    if args.split == 'val':
        score_only_structures = teds.batch_evaluate_html(pred_html_only_structures, gt_html_only_structures)
        score_alls = teds.batch_evaluate_html(pred_html_alls, gt_html_alls)

        print("TEDS score only structure: ", np.mean(score_only_structures))
        print("TEDS score all: ", np.mean(score_alls))
    print("Skipped indices are: ", list(skipped_idx))

    if not args.not_save:
        if args.split == 'val':
            df = pd.DataFrame({
                "img_path": test_img_paths,
                "only_structure_teds_score": score_only_structures,
                "teds_score": score_alls,
                "pred_structure_html": pred_html_only_structures,
                "gt_structure_html": gt_html_only_structures,
                "pred_html": pred_html_alls,
                "gt_html": gt_html_alls,
                "structure_mean_log_prob": max_score_structure_save,
                "cell_mean_log_prob": max_score_cell_mean_save})
        else:
            df = pd.DataFrame({
                "img_path": test_img_paths,
                "pred_structure_html": pred_html_only_structures,
                "pred_html": pred_html_alls,
                "structure_mean_log_prob": max_score_structure_save,
                "cell_mean_log_prob": max_score_cell_mean_save})
        df.to_csv("%s_%s_top_%d_%s_results_SBS_%d_CBS_%d_startIdx_%d_offset_%d.csv" % (args.backbone,
                                                                                       args.EDD_type,
                                                                                       n_samples,
                                                                                       split,
                                                                                       args.beam_size_structure,
                                                                                       args.beam_size_cell,
                                                                                       args.start_idx,
                                                                                       args.offset),
                  index=False)


