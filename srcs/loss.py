import torch
import torch.nn as nn
import torch.nn.functional as F

from postprocess2d import _sigmoid, _tranpose_and_gather_feat

class CustomLoss(nn.Module):
    def __init__(self, device, config, num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.alpha = config['alpha']
        self.beta = config['beta']

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y)  # x_t = x     if label = 1
        # x_t = 1 -x  if label = 0

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * x_t.log()

        return loss.mean()

    def cross_entropy(self, x, y):
        return F.binary_cross_entropy(input=x, target=y, reduction='elementwise_mean')

    def forward(self, preds, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''

        # batch_size = targets.size(0)
        cls_targets, loc_targets = targets.split([1, 6], dim=1)
        if preds.size(1) == 7:
            cls_preds, loc_preds = preds.split([1, 6], dim=1)
        elif preds.size(1) == 15:
            cls_preds, loc_preds, _ = preds.split([1, 6, 8], dim=1)
        ################################################################
        # cls_loss = self.focal_loss(cls_preds, cls_targets)
        ################################################################
        cls_loss = self.cross_entropy(cls_preds, cls_targets) * self.alpha
        cls = cls_loss.item()
        ################################################################
        # reg_loss = SmoothL1Loss(loc_preds, loc_targets)
        ################################################################

        pos_pixels = cls_targets.sum()
        if pos_pixels > 0:
            loc_loss = F.smooth_l1_loss(
                cls_targets * loc_preds, loc_targets, reduction='sum') / pos_pixels * self.beta
            loc = loc_loss.item()
            loss = loc_loss + cls_loss
        else:
            loc = 0.0
            loss = cls_loss

        #print(cls, loc)
        return loss, cls, loc


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
        Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
        Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight


def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class CtdetLoss(torch.nn.Module):
    def __init__(self):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() #if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() #if opt.reg_loss == 'l1' else \
                #RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh =  NormRegL1Loss() #if opt.norm_wh  self.crit_reg

        self.mse_loss = True
        self.hm_weight = 1
        self.wh_weight = 0.1
        self.off_weight = 1

    def forward(self, output, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0

        if not self.mse_loss:
            output['hm'] = _sigmoid(output['hm'])

        hm_loss += self.crit(output['hm'], batch['hm'])

        wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) 
        
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                            batch['ind'], batch['reg'])
            
        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
            self.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                    'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


def test():
    loss = CustomLoss()
    pred = torch.sigmoid(torch.randn(1, 2, 2, 3))
    label = torch.tensor(
        [[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    loss = loss(pred, label)
    print(loss)


if __name__ == "__main__":
    test()
