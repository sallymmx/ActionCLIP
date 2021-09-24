# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler


def to_tuple(x, L):
    if type(x) in (int, float):
        return [x] * L
    if type(x) in (list, tuple):
        if len(x) != L:
            raise ValueError('length of {} ({}) != {}'.format(x, len(x), L))
        return tuple(x)
    raise ValueError('input {} has unkown type {}'.format(x, type(x)))


class WarmupLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        self.num_groups = len(optimizer.param_groups)
        self.warmup_epochs = to_tuple(warmup_epochs, self.num_groups)
        self.warmup_powers = to_tuple(warmup_powers, self.num_groups)
        self.warmup_lrs = to_tuple(warmup_lrs, self.num_groups)
        super(WarmupLR, self).__init__(optimizer, last_epoch)
        assert self.num_groups == len(self.base_lrs)

    def get_lr(self):
        curr_lrs = []
        for group_index in range(self.num_groups):
            if self.last_epoch < self.warmup_epochs[group_index]:
                progress = self.last_epoch / self.warmup_epochs[group_index]
                factor = progress ** self.warmup_powers[group_index]
                lr_gap = self.base_lrs[group_index] - self.warmup_lrs[group_index]
                curr_lrs.append(factor * lr_gap + self.warmup_lrs[group_index])
            else:
                curr_lrs.append(self.get_single_lr_after_warmup(group_index))
        return curr_lrs

    def get_single_lr_after_warmup(self, group_index):
        raise NotImplementedError


class WarmupMultiStepLR(WarmupLR):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):

        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got %s' % repr(milestones))
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer,
                                                warmup_epochs,
                                                warmup_powers,
                                                warmup_lrs,
                                                last_epoch)
        if self.milestones[0] <= max(self.warmup_epochs):
            raise ValueError('milstones[0] ({}) <= max(warmup_epochs) ({})'.format(
                milestones[0], max(self.warmup_epochs)))

    def get_single_lr_after_warmup(self, group_index):
        factor = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return self.base_lrs[group_index] * factor


class WarmupCosineAnnealingLR(WarmupLR):

    def __init__(self,
                 optimizer,
                 total_epoch,
                 final_factor=0,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        self.total_epoch = total_epoch
        self.final_factor = final_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer,
                                                      warmup_epochs,
                                                      warmup_powers,
                                                      warmup_lrs,
                                                      last_epoch)

    def get_single_lr_after_warmup(self, group_index):
        warmup_epoch = self.warmup_epochs[group_index]
        progress = (self.last_epoch - warmup_epoch) / (self.total_epoch - warmup_epoch)
        progress = min(progress, 1.0)
        cosine_progress = (math.cos(math.pi * progress) + 1) / 2
        factor = cosine_progress * (1 - self.final_factor) + self.final_factor
        return self.base_lrs[group_index] * factor


class WarmupExponentialLR(WarmupLR):

    def __init__(self,
                 optimizer,
                 total_epoch,
                 final_factor=1e-3,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        if final_factor <= 0:
            raise ValueError('final_factor ({}) <= 0 not allowed'.format(final_factor))
        self.total_epoch = total_epoch
        self.final_factor = final_factor
        super(WarmupExponentialLR, self).__init__(optimizer,
                                                  warmup_epochs,
                                                  warmup_powers,
                                                  warmup_lrs,
                                                  last_epoch)

    def get_single_lr_after_warmup(self, group_index):
        warmup_epoch = self.warmup_epochs[group_index]
        progress = (self.last_epoch - warmup_epoch) / (self.total_epoch - warmup_epoch)
        progress = min(progress, 1.0)
        factor = self.final_factor ** progress
        return self.base_lrs[group_index] * factor


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
