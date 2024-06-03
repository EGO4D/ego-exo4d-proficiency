import numpy as np
import torch
from torch.nn import functional as F

from .postprocessing import create_frame_dists, apply_hungarian, nonmax_suppress

CONFIG = None


def init_config(cfg):
    global CONFIG
    CONFIG = cfg


@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def construct_hungarian(label_weights):
    opts = CONFIG
    # first split up label_weights
    neg_weight = torch.tensor(label_weights[-1], requires_grad=False).float()
    pos_weight = torch.tensor(label_weights[:-1], requires_grad=False).float()

    if len(opts['devices']) > 0:
        neg_weight = neg_weight.cuda()
        pos_weight = pos_weight.cuda()

    def loss_fn(step, y, y_conv, yhat, pos_mask, neg_mask, frame_mask):
        return hungarian_loss(opts, step, y_conv, yhat, pos_mask, neg_mask, frame_mask, pos_weight, neg_weight)

    return loss_fn


def hungarian_loss(opts, step, y, yhat, pos_mask, neg_mask, mask, pos_weight, neg_weight):
    """Hungarian loss"""
    # figure out the matches.
    TP_weight, FP_weight, num_false_neg, num_false_pos = create_match_array(
        opts, yhat, y, pos_weight, neg_weight)

    seq_len = pos_mask.shape[0]
    mini_batch = pos_mask.shape[1]
    pos_weight = pos_weight.repeat([seq_len, mini_batch, 1])

    pos_mask, neg_mask = create_pos_neg_masks(
        y, pos_weight, neg_weight)
    perframe_cost = perframe_loss(yhat, mask, y, pos_mask, neg_mask)
    tp_cost, fp_cost, fn_cost = structured_loss(
        yhat, mask, pos_weight, TP_weight, FP_weight, num_false_neg)

    total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost =\
        combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost)
    cost = total_cost.mean()

    return cost


def get_perframe_weight(opts, weight, t):
    decay_rate = opts["flags"]["perframe_decay"]
    # decay_step = 36
    if opts["flags"]["anneal_type"] == "exp_step":
        decay_step = opts["flags"]["perframe_decay_step"]
        if opts["flags"]["perframe_decay"] > 0:
            # print(decay_rate**np.floor(t / decay_step))
            # make sure we aren't getting too small...
            if decay_rate**np.floor(t / decay_step) > 0.0000001:
                weight = np.float32(
                    weight * (decay_rate**np.floor(t / decay_step)))
                # prevent underflow?
                weight = max(opts["flags"]["perframe_stop"], weight)
            else:
                weight = opts["flags"]["perframe_stop"]
            # if weight < 0.99:
            #     import pdb; pdb.set_trace()
            # print(weight)
            return weight
    elif opts["flags"]["anneal_type"] == "line_step":
        perframe_start = opts["flags"]["hantman_perframe_weight"]
        perframe_stop = opts["flags"]["perframe_stop"]
        total_epoch_iters = opts["flags"]["total_epochs"] * opts["flags"]["iter_per_epoch"]
        decay_rate = (perframe_stop - perframe_start) / opts["flags"]["total_epochs"]

        weight = weight + decay_rate * np.floor(t / (opts["flags"]["iter_per_epoch"]))

        return weight
    else:
        # print("none")
        return np.float32(weight)


def combine_losses(opts, step, perframe_cost, tp_cost, fp_cost, fn_cost):
    """Combine costs."""
    perframe_lambda = get_perframe_weight(
        opts, opts["flags"]["hantman_perframe_weight"], step)

    # package the weights... needs to be redone otherwise autograd gets confused
    # by repeat variables.... maybe? need to double check.
    tp_lambda = torch.autograd.Variable((torch.Tensor(
        [opts["flags"]["hantman_tp"]]
    )), requires_grad=False).cuda().expand(tp_cost.size())
    fp_lambda = torch.autograd.Variable(torch.Tensor(
        [opts["flags"]["hantman_fp"]]
    ), requires_grad=False).cuda().expand(fp_cost.size())
    fn_lambda = torch.autograd.Variable(torch.Tensor(
        [opts["flags"]["hantman_fn"]]
    ), requires_grad=False).cuda().expand(fn_cost.size())
    perframe_lambda = torch.autograd.Variable(torch.Tensor(
        [float(perframe_lambda)]
    ), requires_grad=False).cuda().expand(perframe_cost.size())
    # struct_lambda = torch.autograd.Variable(torch.Tensor(
    #     [opts["flags"].hantman_struct_weight]
    # ), requires_grad=False).cuda().expand(tp_cost.size())

    # tp_cost = tp_cost * tp_lambda * struct_lambda
    # fp_cost = fp_cost * fp_lambda * struct_lambda
    # fn_cost = fn_cost * fn_lambda * struct_lambda
    # struct_cost = tp_cost + fp_cost + fn_cost
    tp_cost = tp_cost * tp_lambda # * struct_lambda
    fp_cost = fp_cost * fp_lambda # * struct_lambda
    fn_cost = fn_cost * fn_lambda # * struct_lambda
    struct_cost = (tp_cost + fp_cost + fn_cost) * (1 - perframe_lambda)
    # import pdb; pdb.set_trace()

    perframe_cost = perframe_cost * perframe_lambda

    # print("\tperframe, structured: %f, %f" % (perframe_cost.mean().item(), struct_cost.mean().item()))
    total_cost = perframe_cost + struct_cost

    return total_cost, struct_cost, perframe_cost, tp_cost, fp_cost, fn_cost


def structured_loss(predict, frame_mask, label_weight, tp_weight, fp_weight, num_false_neg):
    """Create the structured loss."""
    # tp_weight = torch.autograd.Variable(torch.Tensor(tp_weight), requires_grad=False).cuda()
    # fp_weight = torch.autograd.Variable(torch.Tensor(fp_weight), requires_grad=False).cuda()
    tp_weight = torch.tensor(tp_weight, requires_grad=False).cuda()
    fp_weight = torch.tensor(fp_weight, requires_grad=False).cuda()
    tp_mask = torch.tensor(
        tp_weight > 0, dtype=torch.float32, requires_grad=False).cuda()

    # tp_cost = (tp_mask - tp_weight * predict) * label_weight * frame_mask
    tp_cost = tp_weight * -predict * label_weight * frame_mask
    fp_cost = fp_weight * predict * frame_mask

    tp_cost = tp_cost.sum(2).sum(0).view(-1)
    fp_cost = fp_cost.sum(2).sum(0).view(-1)

    fn_cost = torch.autograd.Variable(
        torch.Tensor(num_false_neg), requires_grad=False).cuda()

    return tp_cost, fp_cost, fn_cost


def perframe_loss(predict, mask, labels, pos_mask, neg_mask):
    """Get the perframe loss."""
    cost = predict * mask - labels * mask
    cost = cost * cost
    cost = (cost * pos_mask + cost * neg_mask)
    cost = cost.sum(2).sum(0).view(-1)
    return cost


def create_match_array(opts, net_out, org_labels, pos_weight, neg_weight):
    """Create the match array."""
    val_threshold = 0.7
    # frame_threshold = [5, 15, 15, 20, 30, 30]
    frame_threshold = [10, 10, 10, 10, 10, 10]
    # frame_threshold = 10
    # y_org = org_labels
    y_org = org_labels.data.cpu().numpy()

    COST_FP = 20
    # COST_FN = 20
    net_out = net_out.data.cpu().numpy()
    num_frames, num_vids, num_classes = net_out.shape
    TP_weight = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
    FP_weight = np.zeros((num_frames, num_vids, num_classes), dtype="float32")
    num_false_neg = []
    num_false_pos = []
    for i in range(num_vids):
        temp_false_neg = 0
        temp_false_pos = 0
        for j in range(num_classes):
            processed, max_vals = nonmax_suppress(
                net_out[:, i, j], val_threshold)
            processed = processed.reshape((processed.shape[0], 1))
            data = np.zeros((len(processed), 3), dtype="float32")
            data[:, 0] = list(range(len(processed)))
            data[:, 1] = processed[:, 0]
            data[:, 2] = y_org[:, i, j]
            # if opts["flags"].debug is True:
            #     import pdb; pdb.set_trace()
            # after suppression, apply hungarian.
            labelled = np.argwhere(y_org[:, i, j] == 1)
            labelled = labelled.flatten().tolist()
            num_labelled = len(labelled)
            dist_mat = create_frame_dists(
                data, max_vals, labelled)
            rows, cols, dist_mat = apply_hungarian(
                dist_mat, frame_threshold[j])

            # missed classifications
            # false_neg = len(labelled) - len(
            #     [k for k in range(len(max_vals)) if cols[k] < len(labelled)])
            # num_false_neg += false_neg
            # temp_false_neg += false_neg

            num_matched = 0
            for pos in range(len(max_vals)):
                ref_idx = max_vals[pos]
                # if cols[pos] < len(labelled):
                row_idx = rows[pos]
                col_idx = cols[pos]
                if col_idx < len(labelled) and\
                        dist_mat[row_idx, col_idx] < frame_threshold[j]:
                    # True positive
                    label_idx = labelled[cols[pos]]
                    # TP_weight[ref_idx, i, j] = np.abs(ref_idx - label_idx)
                    TP_weight[ref_idx, i, j] = 10 - np.abs(ref_idx - label_idx)
                    # if we are reweighting based off label rariety
                    TP_weight[ref_idx, i, j] = TP_weight[ref_idx, i, j] / 10

                    if opts["flags"]["reweight"] is True:
                        # TP_weight[ref_idx, i, j] =\
                        #     TP_weight[ref_idx, i, j] * pos_weigth[j]
                        TP_weight[ref_idx, i, j] =\
                            TP_weight[ref_idx, i, j] * pos_weight[j]
                    num_matched += 1
                else:
                    # False positive
                    FP_weight[ref_idx, i, j] = opts["flags"]["hantman_fp"]
                    if opts["flags"]["reweight"] is True:
                        FP_weight[ref_idx, i, j] =\
                            FP_weight[ref_idx, i, j] * pos_weight[j]
                    temp_false_pos += 1

            temp_false_neg += num_labelled - num_matched
        num_false_neg.append(temp_false_neg)
        num_false_pos.append(temp_false_pos)

    num_false_neg = np.asarray(num_false_neg).astype("float32")

    return TP_weight, FP_weight, num_false_neg, num_false_pos


def create_pos_neg_masks(labels, pos_weight, neg_weight):
    """Create pos/neg masks."""
    # temp = labels.data
    # # import pdb; pdb.set_trace()
    # pos_mask = (temp > 0.9).float() * pos_weight
    # pos_mask = torch.autograd.Variable(pos_mask, requires_grad=False)
    # neg_mask = (temp < 0.001).float() * neg_weight.expand(temp.size())
    # # neg_mask = (temp < 0.9).float() * neg_weight.expand(temp.size())
    # neg_mask = torch.autograd.Variable(neg_mask, requires_grad=False)
    temp = labels.data
    pos_mask = (temp >= 0.7).float() * pos_weight
    pos_mask = torch.autograd.Variable(pos_mask, requires_grad=False)
    neg_mask = (temp < 0.7).float() * neg_weight.expand(temp.size())
    # neg_mask = (temp < 0.9).float() * neg_weight.expand(temp.size())
    neg_mask = torch.autograd.Variable(neg_mask, requires_grad=False)

    return pos_mask, neg_mask
