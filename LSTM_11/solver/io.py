import os
from collections import OrderedDict
import torch


def make_link(filename, link):
    if os.path.islink(link):
        os.remove(link)
    os.symlink(filename, link)


def load_state_dict(module, state_dict, strict=False):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(),
                                           param.size()))
        elif strict:
            raise KeyError(
                'unexpected key "{}" in source state_dict'.format(name))
        else:
            print('ignore key "{}" in source state_dict'.format(name))
    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        if strict:
            raise KeyError(
                'missing keys in source state_dict: "{}"'.format(missing))
        else:
            print('missing keys in source state_dict: "{}"'.format(missing))

def load_state_dict_two(module, state_dict, prefix, strict=False):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        final_name = prefix + '_stage.' + name
        if final_name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[final_name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(final_name, own_state[final_name].size(),
                                           param.size()))
        elif strict:
            raise KeyError(
                'unexpected key "{}" in source state_dict'.format(final_name))
        else:
            print('ignore key "{}" in source state_dict'.format(final_name))



def load_checkpoint(model, filename, strict=False):
    if not os.path.isfile(filename):
        raise IOError('{} is not a checkpoint file'.format(filename))
    checkpoint = torch.load(filename)
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {
            k.lstrip('module.'): v
            for k, v in checkpoint['state_dict'].items()
        }
    if isinstance(model, torch.nn.DataParallel):
        load_state_dict(model.module, state_dict, strict)
    else:
        load_state_dict(model, state_dict, strict)
    return checkpoint['epoch'], checkpoint['num_iters']

def load_checkpoint_two_file(model, pre_filename, post_filename, strict=False):
    if not os.path.isfile(pre_filename):
        raise IOError('{} is not a checkpoint file'.format(pre_filename))
    if not os.path.isfile(post_filename):
        raise IOError('{} is not a checkpoint file'.format(post_filename))
    
    pre_checkpoint = torch.load(pre_filename)
    pre_state_dict = pre_checkpoint['state_dict']
    post_checkpoint = torch.load(post_filename)
    post_state_dict = post_checkpoint['state_dict']

    if isinstance(model, torch.nn.DataParallel):
        load_state_dict_two(model.module, pre_state_dict, 'pre', strict)
        load_state_dict_two(model.module, post_state_dict, 'post', strict)
    else:
        load_state_dict_two(model, pre_state_dict, 'pre', strict)
        load_state_dict_two(model, post_state_dict, 'post', strict)
    return 0, 0


def save_checkpoint(model,
                    epoch,
                    num_iters,
                    out_dir,
                    filename_tmpl='epoch_{}_iter_{}.pth',
                    is_best=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    filename = os.path.join(out_dir, filename_tmpl.format(epoch, num_iters))
    torch.save({
        'epoch': epoch,
        'num_iters': num_iters,
        'state_dict': state_dict_cpu
    }, filename)
    latest_link = os.path.join(out_dir, 'latest.pth')
    make_link(filename, latest_link)
    if is_best:
        best_link = os.path.join(out_dir, 'best.pth')
        make_link(filename, best_link)
