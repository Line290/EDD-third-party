import os
import sys
import torch
from torch import nn
import backbones
import timm
import copy
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.checkpoint import checkpoint
from utils import *

USE_CHECKPOINT=True
USE_CHECKPOINT2=True
USE_CHECKPOINT3=True
FPN=False

class EDD(nn.Module):
    def __init__(self,
                 encoded_image_size=14,
                 encoder_dim=512,   # encoder_dim is abandoned
                 pretrained=True,
                 structure_attention_dim=512,
                 structure_embed_dim=16,
                 structure_decoder_dim=256,
                 structure_dropout=0.5,
                 structure_vocab=None,
                 cell_attention_dim=512,
                 cell_embed_dim=80,
                 cell_decoder_dim=512,
                 cell_dropout=0.2,
                 cell_vocab=None,
                 criterion_structure=None,
                 criterion_cell=None,
                 alpha_c=1.0,
                 id2word_structure=None,
                 id2word_cell=None,
                 last_conv_stride=2,
                 lstm_bias=True,
                 backbone='resnet18'):
        super(EDD, self).__init__()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.encoder = Encoder(
            backbone=backbone,
            encoded_image_size=encoded_image_size,
            encoder_dim=encoder_dim,
            pretrained=pretrained,
            last_stride=last_conv_stride)
        self.encoder_wrapper = ModuleWrapperIgnores2ndArg(self.encoder)

        self.decoder_structure = \
            DecoderStructureWithAttention(
                attention_dim=structure_attention_dim,
                embed_dim=structure_embed_dim,
                decoder_dim=structure_decoder_dim,
                vocab=structure_vocab,
                dropout=structure_dropout,
                encoder_dim=self.encoder.encoder_dim,
                lstm_bias=lstm_bias)

        self.decoder_cell = \
            DecoderCellPerImageWithAttention(
                attention_dim=cell_attention_dim,
                embed_dim=cell_embed_dim,
                decoder_dim=cell_decoder_dim,
                vocab_size=len(cell_vocab),
                dropout=cell_dropout,
                decoder_structure_dim=structure_decoder_dim,
                encoder_dim=self.encoder.encoder_dim,
                lstm_bias=lstm_bias)
        self.criterion_structure = criterion_structure
        self.criterion_cell = criterion_cell
        self.alpha_c = alpha_c
        self.id2word_structure = id2word_structure
        self.id2word_cell = id2word_cell
        self.cell_vocab = cell_vocab
        self.structure_vocab = structure_vocab

    def custom(self, module):
        def custom_forward(*inputs):
            output_1, output_2 = module(inputs[0], inputs[1])
            return output_1, output_2

        return custom_forward

    def forward(self, images,
                caption_structures,
                caplen_structures,
                caption_cells,
                caplen_cells,
                number_cell_per_images):
        if not USE_CHECKPOINT3:
            imgs_structure, imgs_cell = \
                self.encoder_wrapper(images, self.dummy_tensor)
        else:
            imgs_structure, imgs_cell = checkpoint(self.custom(self.encoder_wrapper), images, self.dummy_tensor)

        scores, \
        caps_sorted, \
        decode_lengths_structure, \
        alphas_structure, \
        hidden_states, \
        sort_ind = \
            self.decoder_structure(
                imgs_structure,
                caption_structures,
                caplen_structures)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_structure = \
            pack_padded_sequence(
                scores,
                decode_lengths_structure,
                batch_first=True).data
        targets_structure = \
            pack_padded_sequence(
                targets,
                decode_lengths_structure,
                batch_first=True).data

        loss_structures = \
            self.criterion_structure(
                scores_structure,
                targets_structure)
        loss_structures += \
            self.alpha_c * \
            ((1. - alphas_structure.sum(dim=1)) ** 2).mean()

        if not self.training:
            _, pred_structure = torch.max(scores, dim=2)
            pred_structure = pred_structure.tolist()
            html_trues = list()
            html_predict_only_structures = list()
            html_true_only_structures = list()
            html_predict_only_cells = list()
            html_predict_alls = list()

        # decoder cell per image
        scores_cells_list = []
        target_cells_list = []
        loss_cells = []
        for (ii, ind) in enumerate(sort_ind):
            img = imgs_cell[ind]
            hidden_state_structures = hidden_states[ii]
            hidden_state_structures = torch.stack(hidden_state_structures)
            number_cell_per_image = number_cell_per_images[ind][0]

            caption_cell = caption_cells[ind][:number_cell_per_image]
            caplen_cell = caplen_cells[ind][:number_cell_per_image]

            # Foward encoder image and decoder cell per image
            scores_cell, \
            caps_sorted_cell, \
            decode_lengths_cell, \
            alphas_cell, \
            sort_ind_ = \
                self.decoder_cell(
                    img,
                    caption_cell,
                    caplen_cell,
                    hidden_state_structures,
                    number_cell_per_image)

            if not self.training:
                html_predict_only_structure = ""
                html_true_only_structure = ""
                html_predict_only_cell = ""
                html_true = ""
                html_predict_all = ""

                _, pred_cells = torch.max(scores_cell, dim=2)
                pred_cells = pred_cells.tolist()
                ground_truth = list()

                # get cell content in per images when predict
                temp_preds = [''] * len(pred_cells)
                for j, p in enumerate(pred_cells):
                    # because sort cell with descending, mapping pred_cell to sort_ind_
                    words = p[:decode_lengths_cell[j]]
                    temp_preds[sort_ind_[j]] += convertId2wordSentence(self.id2word_cell, words)

                # get cell content in per images ground_truth
                for j in range(caption_cell.shape[0]):
                    img_caps = caption_cell[j].tolist()
                    img_captions = [w for w in img_caps if w not in {
                        self.cell_vocab['<start>'], self.cell_vocab['<pad>']}]  # remove <start> and pads
                    ground_truth.append(convertId2wordSentence(
                        self.id2word_cell, img_captions))

                index_cell = 0
                cap_structure = caps_sorted[ii][:decode_lengths_structure[ii]].tolist()
                pred_structure_image = pred_structure[ii][:decode_lengths_structure[ii]]

                for (index, c) in enumerate(cap_structure):
                    if c == self.structure_vocab["<start>"] or c == self.structure_vocab["<end>"]:
                        continue
                    html_predict_only_cell += self.id2word_structure[c]
                    html_predict_only_structure += self.id2word_structure[pred_structure_image[index - 1]]
                    html_true_only_structure += self.id2word_structure[c]
                    html_true += self.id2word_structure[c]
                    html_predict_all += self.id2word_structure[pred_structure_image[index - 1]]
                    if c == self.structure_vocab["<td>"] or c == self.structure_vocab[">"]:
                        html_predict_only_cell += temp_preds[index_cell]
                        html_true += ground_truth[index_cell]
                        html_predict_all += temp_preds[index_cell]
                        index_cell += 1

                html_predict_only_structure_ = create_html(html_predict_only_structure)
                html_true_only_structure_ = create_html(html_true_only_structure)
                html_predict_only_cell_ = create_html(html_predict_only_cell)
                html_predict_all_ = create_html(html_predict_all)
                html_true_ = create_html(html_true)

                html_predict_only_structures.append(html_predict_only_structure_)
                html_true_only_structures.append(html_true_only_structure_)
                html_predict_only_cells.append(html_predict_only_cell_)
                html_predict_alls.append(html_predict_all_)
                html_trues.append(html_true_)

            target_cell = caps_sorted_cell[:, 1:]
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_cell = pack_padded_sequence(
                scores_cell, decode_lengths_cell, batch_first=True).data
            target_cell = pack_padded_sequence(
                target_cell, decode_lengths_cell, batch_first=True).data
            scores_cells_list.append(scores_cell)
            target_cells_list.append(target_cell)
            loss_cell = self.criterion_cell(scores_cell, target_cell)
            loss_cell += self.alpha_c * ((1. - alphas_cell.sum(dim=1)) ** 2).mean()
            loss_cells.append(loss_cell)

        scores_cell = torch.cat(scores_cells_list, dim=0)
        targets_cell = torch.cat(target_cells_list, dim=0)
        loss_cells = torch.stack(loss_cells)
        loss_cells = torch.mean(loss_cells)

        if self.training:
            return loss_structures, scores_structure, targets_structure, \
                   loss_cells, scores_cell, targets_cell
        else:
            return loss_structures, loss_cells, \
                   html_predict_only_structures, \
                   html_true_only_structures, \
                   html_predict_only_cells, \
                   html_predict_alls, \
                   html_trues, \
                   scores_structure, targets_structure, \
                   scores_cell, targets_cell


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self,
                 backbone='resnet18',
                 encoded_image_size=28//2,
                 encoder_dim=512,
                 pretrained=True,
                 last_stride=2):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.backbone = backbone

        if 'resnet' in backbone or 'resnext' in backbone:
            backbone_net = getattr(backbones, backbone)(pretrained=pretrained, last_stride=last_stride)
            if backbone == 'resnet18' or backbone == 'resnet34':
                self.encoder_dim = encoder_dim
            elif backbone == 'resnet50':
                self.encoder_dim = encoder_dim * 4
            elif backbone == 'resnext101_32x8d':
                print(backbone)
                self.encoder_dim = encoder_dim
                self.downsample_conv_structure = nn.Conv2d(2048, self.encoder_dim, kernel_size=1, stride=1, bias=False)
                self.downsample_bn_structure = nn.BatchNorm2d(self.encoder_dim)
                self.downsample_conv_cell = nn.Conv2d(2048, self.encoder_dim, kernel_size=1, stride=1, bias=False)
                self.downsample_bn_cell = nn.BatchNorm2d(self.encoder_dim)
                self.relu = nn.ReLU(inplace=True)

            offset = -3
            backbone_net_all_layer = list(backbone_net.children())
            modules = backbone_net_all_layer[:offset]
            self.last_conv_block_for_structure = backbone_net_all_layer[offset]
            self.last_conv_block_for_cell = copy.deepcopy(self.last_conv_block_for_structure)

            self.backbone_net = nn.Sequential(*modules)
        else:
            backbone_net = timm.create_model(backbone,
                                             pretrained=pretrained,
                                             features_only=True,
                                             out_indices=(4,),
                                             output_stride=last_stride*16)

            self.encoder_dim = backbone_net.feature_info.get(key='num_chs', idx=4)
            offset = -1
            backbone_net_all_layer = list(backbone_net.children())
            modules = backbone_net_all_layer[:offset]
            modules.extend(backbone_net_all_layer[offset][:-2])
            self.last_conv_block_for_structure = backbone_net_all_layer[offset][-2:]
            self.last_conv_block_for_cell = copy.deepcopy(self.last_conv_block_for_structure)
            self.backbone_net = nn.Sequential(*modules)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.backbone_net(
            images)  # (batch_size, encoder_dim, image_size/16, image_size/16)，

        # (batch_size, encoder_dim, encoded_image_size, encoded_image_size)
        out_for_structure = self.last_conv_block_for_structure(out)
        if self.backbone == 'resnext101_32x8d':
            out_for_structure = self.downsample_conv_structure(out_for_structure)
            out_for_structure = self.downsample_bn_structure(out_for_structure)
            out_for_structure = self.relu(out_for_structure)
        out_for_structure = out_for_structure.permute(0, 2, 3, 1)

        out_for_cell = self.last_conv_block_for_cell(out)
        if self.backbone == 'resnext101_32x8d':
            out_for_cell = self.downsample_conv_cell(out_for_cell)
            out_for_cell = self.downsample_bn_cell(out_for_cell)
            out_for_cell = self.relu(out_for_cell)
        out_for_cell = out_for_cell.permute(0, 2, 3, 1)

        # (batch_size, encoded_image_size, encoded_image_size, encoder_dim)
        return out_for_structure, out_for_cell


class Soft_Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim, structure_decoder_dim=None):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Soft_Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        if structure_decoder_dim is not None:
            self.structure_decoder_att = nn.Linear(structure_decoder_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden, structure_hidden=None):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        att1 = self.encoder_att(
            encoder_out)  # (batch_size, num_pixels, attention_dim)

        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        # attention is created by output encoder and previous decoder output
        # size (batch_size, num_pixels)
        if structure_hidden is not None:
            att3 = self.structure_decoder_att(structure_hidden) # (batch_size, attention_dim)
            att = self.full_att(self.relu(att1 + att3.unsqueeze(1)+ att2.unsqueeze(1))).squeeze(2)
        else:
            att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)  # size is (batch_size, num_pixels)
        attention_weighted_encoding = (
            encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # size is (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderStructureWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab, encoder_dim=512, dropout=0.5, lstm_bias=True):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderStructureWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab = vocab
        self.id2words = id_to_word(vocab)
        self.vocab_size = len(vocab)
        self.dropout = dropout

        self.attention = Soft_Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(
            self.vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=lstm_bias)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, self.vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def custom(self, module):
        def custom_forward(*inputs):
            output1, output2 = module(inputs[0], inputs[1])
            return output1, output2

        return custom_forward

    def run_function(self, module):
        def custom_forward(*inputs):
            output, hidden = module(
                inputs[0], (inputs[1], inputs[2])
            )
            return output, hidden

        return custom_forward

    def custom_2(self, module):
        def custom_forward(inputs):
            output = module(inputs)
            return output

        return custom_forward

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # bs, 28*28, 512
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]

        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim) # 16
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM structure state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(encoder_out.device)

        # create hidden_states to generate cell
        hidden_states = [[] for x in range(batch_size)]

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        # using teacher forcing, save hidden state h_k+1 if ground truth t_k is <td> or >
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if not USE_CHECKPOINT:
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
            else:
                attention_weighted_encoding, alpha = checkpoint(self.custom(self.attention),
                                                                encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            if not USE_CHECKPOINT2:
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            else:
                gate = checkpoint(self.custom_2(self.f_beta), h[:batch_size_t])
                gate = checkpoint(self.custom_2(self.sigmoid), gate)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # hidden h_t+1 and ground_truth token t_t
            if not USE_CHECKPOINT2:
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :],
                               attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h, c = checkpoint(self.run_function(self.decode_step),
                                  torch.cat([embeddings[:batch_size_t, t, :],
                                             attention_weighted_encoding],
                                            dim=1),
                                  h[:batch_size_t], c[:batch_size_t])

            # get and save hidden state h_k+1 when groun_truth token in t_k is <td> or >
            for i in range(batch_size_t):
                if self.vocab["<td>"] == encoded_captions[i][t].cpu().numpy() or self.vocab[">"] == encoded_captions[i][t].cpu().numpy():
                    hidden_states[i].append(h[i])

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, hidden_states, sort_ind


class DecoderCellPerImageWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, decoder_structure_dim, vocab_size, encoder_dim=512, dropout=0.5, lstm_bias=True):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderCellPerImageWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Soft_Attention(
            encoder_dim, decoder_dim, attention_dim, decoder_structure_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=lstm_bias)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def custom(self, module):
        def custom_forward(*inputs):
            output1, output2 = module(inputs[0], inputs[1], inputs[2])
            return output1, output2

        return custom_forward

    def run_function(self, module):
        def custom_forward(*inputs):
            output, hidden = module(
                inputs[0], (inputs[1], inputs[2])
            )
            return output, hidden

        return custom_forward

    def custom_2(self, module):
        def custom_forward(inputs):
            output = module(inputs)
            return output

        return custom_forward

    def forward(self, encoder_out, encoded_captions, caption_lengths, hidden_state_structures, batch_size):

        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # expand encoder_out (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(-1, encoder_dim)
        encoder_out = encoder_out.squeeze(1).repeat(batch_size, 1, 1)

        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.unsqueeze(
            1).squeeze(
            1).sort(dim=0, descending=True)
        # caption_lengths, sort_ind = caption_lengths.sort(descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # hidden_state_structures size (batch_size, decoder_dim)
        hidden_state_structures = hidden_state_structures[sort_ind]
        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM cell state
        # size is (batch_size, decoder_dim)
        h, c = self.init_hidden_state(encoder_out)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(encoder_out.device)

        # decode with time step
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if not USE_CHECKPOINT:
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t],
                                                                    hidden_state_structures[:batch_size_t])
            else:
                attention_weighted_encoding, alpha = checkpoint(self.custom(self.attention),
                                                                encoder_out[:batch_size_t],
                                                                h[:batch_size_t],
                                                                hidden_state_structures[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            if not USE_CHECKPOINT2:
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            else:
                gate = checkpoint(self.custom_2(self.f_beta), h[:batch_size_t])
                gate = checkpoint(self.custom_2(self.sigmoid), gate)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # concat hidden state structure + attention_weighted_encoding
            # attention_weighted_encoding = torch.cat(
            #     (attention_weighted_encoding, hidden_state_structures[:batch_size_t]), dim=1)
            # hidden h_t+1 and ground_truth token t_t
            if not USE_CHECKPOINT2:
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :],
                               attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            else:
                h, c = checkpoint(self.run_function(self.decode_step),
                                  torch.cat([embeddings[:batch_size_t, t, :],
                                             attention_weighted_encoding],
                                            dim=1),
                                  h[:batch_size_t], c[:batch_size_t])

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
