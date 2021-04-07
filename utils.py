import os
import numpy as np
import json
import torch
import torch.distributed as dist

from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
import random
import jsonlines
from bs4 import BeautifulSoup as bs
from html import escape


def check_longest_cell(cells):
    length_cells = [len(cell["tokens"]) for cell in cells]
    return max(length_cells)


def create_input_files(image_folder="pubtabnet", output_folder="output_w_none_399k",
                       max_len_token_structure=300,
                       max_len_token_cell=100,
                       image_size=512
                       ):
    """
    Creates input files for training, validation, and test data.

    :param image_folder: folder with downloaded images
    :param output_folder: folder to save files
    :param max_len_token_structure: don't sample captions_structure longer than this length
    :param max_len_token_cell: sample captions_cell longer than this length will be clipped
    """
    print("create_input .....")
    with open(os.path.join(image_folder, "PubTabNet_2.0.0.jsonl"), 'r') as reader:
        imgs = list(reader)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read image paths and captions for each image
    train_image_captions_structure = []
    train_image_captions_cells = []
    train_image_paths = []

    valid_image_captions_structure = []
    valid_image_captions_cells = []
    valid_image_paths = []

    test_image_captions_structure = []
    test_image_captions_cells = []
    test_image_paths = []
    word_freq_structure = Counter()
    word_freq_cells = Counter()

    max_number_imgs_train = 100000000
    max_number_imgs_val = 1000000

    total_number_imgs_train = 0
    total_number_imgs_val = 0
    total_number_imgs_test = 0

    for (index, image) in tqdm(enumerate(imgs)):
        img = eval(image)
        word_freq_structure.update(img["html"]["structure"]["tokens"])

        for cell in img["html"]["cells"]:
            if len(cell["tokens"]) == 0:   # an empty cell
                cell["tokens"].append('<None>')
            word_freq_cells.update(cell["tokens"])

        captions_structure = []
        caption_cells = []
        path = os.path.join("{}/{}".format(image_folder,
                                           img["split"]), img['filename'])

        captions_structure.append(img["html"]["structure"]['tokens'])  # List

        if img["split"] == "train" and total_number_imgs_train < max_number_imgs_train:
            if len(img["html"]["structure"]["tokens"]) <= max_len_token_structure:
                # img_pic = imread(path)
                # if img_pic.shape[0] <= image_size and img_pic.shape[1] <= image_size:
                for cell in img["html"]["cells"]:
                    caption_cells.append(cell["tokens"][:max_len_token_cell])
                train_image_captions_structure.append(captions_structure)  # List[List]
                train_image_captions_cells.append(caption_cells)  # List[List[List]]
                train_image_paths.append(path)
                total_number_imgs_train += 1
            else:
                continue
        elif img["split"] == "val" and total_number_imgs_val < max_number_imgs_val:
            if len(img["html"]["structure"]["tokens"]) <= max_len_token_structure:
                for cell in img["html"]["cells"]:
                    caption_cells.append(cell["tokens"][:max_len_token_cell])
                valid_image_captions_structure.append(captions_structure)
                valid_image_captions_cells.append(caption_cells)
                valid_image_paths.append(path)
                total_number_imgs_val += 1
        elif img["split"] == "test":
            test_image_captions_structure.append(captions_structure)
            test_image_captions_cells.append(caption_cells)
            test_image_paths.append(path)
            total_number_imgs_test += 1
        else:
            continue
    print("Total number imgs for train: ", total_number_imgs_train)
    print("Total number imgs for val: ", total_number_imgs_val)
    print("Total number imgs for test: ", total_number_imgs_test)

    # create vocabluary structure
    words_structure = [w for w in word_freq_structure.keys()]
    word_map_structure = {k: v + 1 for v, k in enumerate(words_structure)}
    word_map_structure['<unk>'] = len(word_map_structure) + 1
    word_map_structure['<start>'] = len(word_map_structure) + 1
    word_map_structure['<end>'] = len(word_map_structure) + 1
    word_map_structure['<pad>'] = 0

    # create vocabluary cells
    words_cell = [w for w in word_freq_cells.keys()]
    word_map_cell = {k: v + 1 for v, k in enumerate(words_cell)}
    word_map_cell['<unk>'] = len(word_map_cell) + 1
    word_map_cell['<start>'] = len(word_map_cell) + 1
    word_map_cell['<end>'] = len(word_map_cell) + 1
    word_map_cell['<pad>'] = 0

    # save vocabluary to json
    with open(os.path.join(output_folder, 'WORDMAP_' + "STRUCTURE" + '.json'), 'w') as j:
        json.dump(word_map_structure, j)

    with open(os.path.join(output_folder, 'WORDMAP_' + "CELL" + '.json'), 'w') as j:
        json.dump(word_map_cell, j)

    for impaths, imcaps_structure, imcaps_cell, split in [(train_image_paths,
                                                           train_image_captions_structure,
                                                           train_image_captions_cells,
                                                           'train'),
                                                          (valid_image_paths,
                                                           valid_image_captions_structure,
                                                           valid_image_captions_cells,
                                                           'val'),
                                                          (test_image_paths,
                                                           test_image_captions_structure,
                                                           test_image_captions_cells,
                                                           'test')]:

        if len(imcaps_structure) == 0 and split in ['train', 'val']:
            continue

        with open(os.path.join(output_folder, split + '_IMAGE_PATHS.txt'), 'a') as f:
            print("\nReading %s images and captions, storing to file...\n" % split)
            enc_captions_structure = []
            enc_captions_cells = []
            cap_structure_len = []
            cap_cell_len = []
            number_cell_per_images = []
            for i, path in enumerate(tqdm(impaths)):
                captions_structure = imcaps_structure[i]
                captions_cell = imcaps_cell[i]
                f.write(impaths[i]+'\n')

                # encode caption cell and structure
                for j, c in enumerate(captions_structure):
                    enc_c = [word_map_structure['<start>']] + \
                            [word_map_structure.get(word, word_map_structure['<unk>']) for word in c] + \
                            [word_map_structure['<end>']]
                    c_len = len(c) + 2
                    enc_captions_structure.append(enc_c)
                    cap_structure_len.append(c_len)

                # for each img have many cell captions
                each_enc_captions_cell = []
                each_cap_cell_len = []
                for j, c in enumerate(captions_cell):
                    enc_c = [word_map_cell['<start>']] + \
                            [word_map_cell.get(word, word_map_cell['<unk>']) for word in c] + \
                            [word_map_cell['<end>']]
                    c_len = len(c) + 2
                    each_enc_captions_cell.append(enc_c)
                    each_cap_cell_len.append(c_len)

                # save encoding cell in per image
                enc_captions_cells.append(each_enc_captions_cell)
                cap_cell_len.append(each_cap_cell_len)
                number_cell_per_images.append(len(captions_cell))
            if split == 'train' or split == 'val':
                with open(os.path.join(output_folder, split + '_CAPTIONS_STRUCTURE' + '.json'), 'w') as j:
                    json.dump(enc_captions_structure, j)
                with open(os.path.join(output_folder, split + '_CAPLENS_STRUCTURE' + '.json'), 'w') as j:
                    json.dump(cap_structure_len, j)
                with open(os.path.join(output_folder, split + '_CAPTIONS_CELL' + '.json'), 'w') as j:
                    json.dump(enc_captions_cells, j)
                with open(os.path.join(output_folder, split + '_CAPLENS_CELL' + '.json'), 'w') as j:
                    json.dump(cap_cell_len, j)
                with open(os.path.join(output_folder, split + '_NUMBER_CELLS_PER_IMAGE' + '.json'), 'w') as j:
                    json.dump(number_cell_per_images, j)


def id_to_word(vocabluary):
    id2word = {value: key for key, value in vocabluary.items()}
    return id2word


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(
            lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                if param.grad.data.max().item() > grad_clip or param.grad.data.min().item() < -grad_clip:
                    print("Clip gradient......")
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_tmp_grad(optimizer, filename):
    save_list = []
    for group in optimizer.param_groups:
        # torch.save([(p, p.grad.data) for p in group['params'] if p.grad is not None], filename)

    # torch.save([p for p in model.parameters() if p.requires_grad], "model_"+filename)
        for param in group['params']:
            if param.grad is not None:
                print(param.name, param.grad.data.size())
                save_list.append(param.grad.data.cpu().numpy())
    # torch.save(save_list, filename)
    np.save(filename, np.array(save_list))


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder_structure, decoder_cell,
                    encoder_optimizer, decoder_structure_optimizer, decoder_cell_optimizer, recent_ted_score, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param recent_ted_score: validation TED score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'ted_score': recent_ted_score,
             'encoder': encoder,
             'decoder_structure': decoder_structure,
             'encoder_optimizer': encoder_optimizer,
             'decoder_structure_optimizer': decoder_structure_optimizer,
             'decoder_cell': decoder_cell,
             'decoder_cell_optimizer': decoder_cell_optimizer,
             }
    filename = 'checkpoint_table' + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


def create_html(html_code):
    return '''<html>
                   <head>
                   <meta charset="UTF-8">
                   <style>
                   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
                   }
                   </style>
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%%">
                     %s
                   </table>
                   </body>
                   </html>''' % html_code


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


def adjust_learning_rate(optimizer, lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param lr: new lr.
    """

    # print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # optimizer.state_dict()['param_groups'][0]['lr'] = lr
    # print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    :return List[Tensor]: in order to all reduce
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k * (100.0 / batch_size))
        return res


def format_html(html):
    ''' Formats HTML code from tokenized annotation of img
    '''

    html_code = '''<html>
                   <head>
                   <meta charset="UTF-8">
                   <style>
                   table, th, td {
                     border: 1px solid black;
                     font-size: 10px;
                   }
                   </style>
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%%">
                     %s
                   </table>
                   </body>
                   </html>''' % html

    # prettify the html
    soup = bs(html_code)
    html_code = soup.prettify()
    return html_code


def convertId2wordSentence(id2word, idwords):
    words = [id2word[idword] for idword in idwords]
    words = [word for word in words if word != "<end>" and word != "<start>" and word != "<None>"]
    words = "".join(words)
    return words


def mean_loss(loss_list):
    loss_tensor = torch.stack(loss_list)
    loss_tensor_mean = torch.mean(loss_tensor)
    return loss_tensor_mean


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(epoch, epochs_since_improvement, model,
                   optimizer, recent_ted_score, is_best, filename):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param recent_ted_score: validation TED score for this epoch
    :param is_best: is this checkpoint the best so far?
    :param filename: checkpoint's name
    """
    if is_main_process():
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'ted_score': recent_ted_score,
                 'model': model.module.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 }
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            best_filename = filename.split('/')
            best_filename[-1] = 'BEST_' + best_filename[-1]
            best_filename = '/'.join(best_filename)
            torch.save(state, best_filename)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
