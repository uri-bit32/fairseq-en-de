# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import pdb
import torch.nn.functional as F



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
#    pdb.set_trace()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.teacher_label = None
        self.teacher_annel_start_step = 10000
        self.teacher_annel_end_step = 20000
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
#        pdb.set_trace()
        mode = 1
        if mode == 0:
           sample['net_input']['index'] = 12
           net_output_max = model(**sample['net_input'])
           loss_max, nll_loss_max, lprobs_max, detach_probs_max, target = self.compute_loss(model, net_output_max, sample, reduce=reduce)

           index = int(torch.LongTensor(1).random_(1, 12))
           sample['net_input']['index'] = index
           net_output_random = model(**sample['net_input'])
           loss_random, nll_loss_random, lprobs_random, detach_probs_random, target = self.compute_loss(model, net_output_random, sample, reduce=reduce)

           sample['net_input']['index'] = 0
           net_output_min = model(**sample['net_input'])
           loss_min, nll_loss_min, lprobs_min, detach_probs_min, target = self.compute_loss(model, net_output_min, sample, reduce=reduce)

           loss = loss_max + loss_random + loss_min
           nll_loss = nll_loss_max + nll_loss_random + nll_loss_min

           detach_ensemble_teacher = (detach_probs_max + detach_probs_random + detach_probs_min) / 3
           teacher_loss = -(detach_ensemble_teacher * lprobs_max).sum() - (detach_ensemble_teacher * lprobs_random).sum() -  (detach_ensemble_teacher * detach_probs_min).sum()

           loss = loss + teacher_loss

           sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
           logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
           }
        else:
           net_output = model(**sample['net_input'])
           loss, nll_loss  = self.compute_loss_distillation_v2(model, net_output, sample, reduce=reduce)
           sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
           logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
           }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        detach_lprobs = model.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1)).detach()
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss, lprobs, detach_lprobs, target

    def compute_loss_distillation_v2(self, model, net_output, sample, reduce=True):
#        pdb.set_trace()
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        if model.training:
           if sample['net_input']['index'] == 12:
              self.teacher_label = None
              self.teacher_label = model.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1)).detach()
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        stage = 0
        if stage == 0:
           alpha = 0
        else:
           alpha = 0.5
        if model.training:
           if sample['net_input']['index'] != 12:
              teacher_loss = -(self.teacher_label * lprobs).sum()
              loss = loss * (1 - alpha) + teacher_loss * alpha     
        return loss, nll_loss


    def compute_loss_distillation(self, model, net_output, sample, reduce=True):
#        pdb.set_trace()
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        if model.training:
           if sample['net_input']['index'] != 0:
              if sample['net_input']['index']!=12:
                 self.small_teacher_label = None
                 self.small_teacher_label = model.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1)).detach()
              else:
                 self.teacher_label = None
                 self.teacher_label = model.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1)).detach()
####              sample['net_input']['ema_mode'] = 1
####              ema_net_output = model(**sample['net_input'])
  
####              sample['net_input']['ema_mode'] = 0
#              self.teacher_label = model.get_normalized_probs(net_output, log_probs=False).view(-1, lprobs.size(-1)).detach()
###           else:
#              target = self.teacher_label.argmax(dim=1).unsqueeze(1)
####              target = model.get_targets(sample, net_output).view(-1, 1)
#              self.teacher_label = None
####        else:
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        
        if model.training:
           if sample['net_input']['index'] != 12:
              if sample['net_input']['index'] == 0:
                 teacher_loss = -(self.small_teacher_label * lprobs).sum()
                 loss = loss * 0.5 + teacher_loss * 0.5
              else:
                 teacher_loss = -(self.teacher_label * lprobs).sum()
                 loss = loss * 0.5 + teacher_loss * 0.5
              
####              if sample['step'] < self.teacher_annel_start_step:
####                 alpha = 0
####              elif sample['step'] > self.teacher_annel_end_step:
####                 alpha = 1
####              else:
####                 ratio  = (sample['step'] - self.teacher_annel_start_step) / (self.teacher_annel_end_step - self.teacher_annel_start_step)
####                 alpha = 1 * ratio
####              teacher_loss = -(self.teacher_label * lprobs).sum()
#              teacher_loss = F.kl_div(lprobs, self.teacher_label, size_average=False) 
####              alpha = 1
####              loss = loss * (1 - alpha) + teacher_loss * alpha
 
        return loss, nll_loss



    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_sum_random = sum(log.get('loss_random', 0) for log in logging_outputs)
        loss_sum_max = sum(log.get('loss_max', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))
        
        metrics.log_scalar('loss_random', loss_sum_random / sample_size / math.log(2), sample_size, round=4)
        metrics.log_scalar('loss_max', loss_sum_max / sample_size / math.log(2), sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
