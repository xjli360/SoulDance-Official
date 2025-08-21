# Copyright (c) Bytedance

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Copyright (c) [2025] [Bytedance]
# Copyright (c) [2025] [Xiaojie Li] 
# This file has been modified by Xiaojie Li on 2025/07/30

import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
# from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from models.vq.quantizer import QuantizeEMAReset, QuantizeEMA

from einops import rearrange, repeat, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class
class EnhancedTransformationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(EnhancedTransformationLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        transformed = self.linear(x)
        attn_output, _ = self.self_attention(transformed, transformed, transformed)
        x = self.norm1(transformed + attn_output)
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)

        return output


class TransformationLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformationLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size, input_dim, seq = x.shape

        x = x.permute(0,2,1).reshape(-1, input_dim)
        
        # 2. Apply the linear transformation
        transformed = self.linear(x)  # (batch_size * sequence_length, output_dim)
        
        # 3. Reshape back to the original batch and sequence dimensions
        transformed = transformed.reshape(batch_size, -1, seq)
        return transformed




class HiResidualVQ(nn.Module):
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        # self.layers = nn.ModuleList([VectorQuantize(accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])
        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers_body = nn.ModuleList([layer for _ in range(num_quantizers)])
            self.layers_hands = nn.ModuleList([layer for _ in range(num_quantizers)])
            self.layers_face = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers_body = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
            self.layers_hands = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
            self.layers_face = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        # self.layers = nn.ModuleList([QuantizeEMA(**kwargs) for _ in range(num_quantizers)])

        # self.quantize_dropout = quantize_dropout and num_quantizers > 1
        self.code_dim = kwargs['code_dim']
        self.transformation = TransformationLayer(self.code_dim, self.code_dim)
        self.conv1d = nn.Conv1d(in_channels=self.code_dim * 2, out_channels=self.code_dim, kernel_size=3, stride=1, padding=1)

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob
            
    @property
    def codebooks(self):
        codebooks_b = [layer.codebook for layer in self.layers_body]
        codebooks_b = torch.stack(codebooks_b, dim = 0)

        codebooks_h = [layer.codebook for layer in self.layers_hands]
        codebooks_h = torch.stack(codebooks_h, dim = 0)

        codebooks_f = [layer.codebook for layer in self.layers_face]
        codebooks_f = torch.stack(codebooks_f, dim = 0)
        codebooks_whole = [codebooks_b, codebooks_h, codebooks_f]
        return codebooks_whole # 'q c d'
    
    def get_codes_from_indices(self, indices_whole): #indices shape 'b n num_quantizers' # dequantize  # 256 16 5
        # import pdb;pdb.set_trace()
        n = indices_whole.shape[1] // 3
        all_codes_whole_list = []
        for idx in range(3):
            indices = indices_whole[:, n*idx:n*(idx+1), :]
            batch, quantize_dim = indices.shape[0], indices.shape[-1]
            # because of quantize dropout, one can pass in indices that are coarse
            # and the network should be able to reconstruct

            if quantize_dim < self.num_quantizers:
                indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

            # get ready for gathering
            codebooks = repeat(self.codebooks[idx], 'q c d -> q b c d', b = batch) # torch.Size([1, 1024, 6])
            gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

            # take care of quantizer dropout

            mask = gather_indices == -1.
            gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

            # print(gather_indices.max(), gather_indices.min())
            all_codes = codebooks.gather(2, gather_indices) # gather all codes

            # mask out any codes that were dropout-ed

            all_codes = all_codes.masked_fill(mask, 0.)
            all_codes_whole_list.append(all_codes)

        # import pdb;pdb.set_trace()
        all_codes_whole = torch.cat(all_codes_whole_list, dim=3) 
        return all_codes_whole # 'q b n d' (q, batch, 512 * 3, 16)

    def get_codebook_entry(self, indices): #indices shape 'b n q'  # TODO
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, sample_codebook_temp = None, force_dropout_index=-1):
        # debug check
        # print(self.codebooks[:,0,0].detach().cpu().numpy())
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x[0].device

        quantized_out_body, quantized_out_hands, quantized_out_face = 0., 0., 0.
        residual_body,residual_hands,residual_face = x[0],x[1],x[2]

        all_losses = []
        all_indices = []
        all_perplexity = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        # To ensure the first-k layers learn things as much as possible, we randomly dropout the last q - k layers
        null_indices_shape = [x[0].shape[0], x[0].shape[-1]]
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) # keep quant layers <= quantize_dropout_cutoff_index, TODO vary in batch
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            # null_loss = 0.

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        # print(force_dropout_index)
        # go through the layers

        for quantizer_index in range(self.num_quantizers):

            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long).repeat(1, 3)
                all_indices.append(null_indices)
                # all_losses.append(null_loss)
                continue

            # layer_indices = None
            # if return_loss:
            #     layer_indices = indices[..., quantizer_index] #gt indices

            # quantized, *rest = layer(residual, indices = layer_indices, sample_codebook_temp = sample_codebook_temp) #single quantizer TODO
            # import pdb;pdb.set_trace()
            quantized_body, *rest_body = self.layers_body[quantizer_index](residual_body, return_idx=True, temperature=sample_codebook_temp) #single quantizer  256, 512, 16 -> 256, 512, 16
            residual_body -= quantized_body.detach()
            quantized_out_body += quantized_body

            hiera_hands = self.conv1d(torch.cat((self.transformation(quantized_body), residual_hands), dim=1))
            quantized_hands, *rest_hands = self.layers_hands[quantizer_index](hiera_hands, return_idx=True, temperature=sample_codebook_temp)
            residual_hands -= quantized_hands.detach()
            quantized_out_hands += quantized_hands

            hiera_faces = self.conv1d(torch.cat((self.transformation(quantized_hands), residual_face), dim=1))
            quantized_face, *rest_face = self.layers_face[quantizer_index](hiera_faces, return_idx=True, temperature=sample_codebook_temp)
            residual_face -= quantized_face.detach()
            quantized_out_face += quantized_face

            embed_indices, loss, perplexity = torch.cat((rest_body[0], rest_hands[0], rest_face[0]), dim=-1), rest_body[1]+rest_hands[1]+rest_face[1], rest_body[2]+rest_hands[2]+rest_face[2]
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        # import pdb;pdb.set_trace()
        quantized_out = torch.cat((quantized_out_body, quantized_out_hands, quantized_out_face), dim=1)
        # stack all losses and indices
        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses)/len(all_losses)
        all_perplexity = sum(all_perplexity)/len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret
    
    def quantize(self, x, return_latent=False):
        # quantized_out = 0.
        # residual = x
        quantized_out_body, quantized_out_hands, quantized_out_face = 0., 0., 0.
        residual_body,residual_hands,residual_face = x[0],x[1],x[2]
        all_codes = []
        all_indices = []
        for quantizer_index in range(self.num_quantizers):
            quantized_body, *rest_body = self.layers_body[quantizer_index](residual_body, return_idx=True) #single quantizer  256, 512, 16 -> 256, 512, 16
            residual_body -= quantized_body.detach()
            quantized_out_body += quantized_body

            hiera_hands = self.conv1d(torch.cat((self.transformation(quantized_body), residual_hands), dim=1))
            quantized_hands, *rest_hands = self.layers_hands[quantizer_index](hiera_hands, return_idx=True)
            residual_hands -= quantized_hands.detach()
            quantized_out_hands += quantized_hands

            hiera_faces = self.conv1d(torch.cat((self.transformation(quantized_hands), residual_face), dim=1))
            quantized_face, *rest_face = self.layers_face[quantizer_index](hiera_faces, return_idx=True)
            residual_face -= quantized_face.detach()
            quantized_out_face += quantized_face

            embed_indices, loss, perplexity = torch.cat((rest_body[0], rest_hands[0], rest_face[0]), dim=-1), rest_body[1]+rest_hands[1]+rest_face[1], rest_body[2]+rest_hands[2]+rest_face[2]
            all_indices.append(embed_indices)
            quantized = torch.cat((quantized_body, quantized_hands, quantized_face), dim=1)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx