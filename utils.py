from __future__ import print_function, division, absolute_import

import os
import shutil
import visdom
from glob import glob
import time
import pickle

import torch
import numpy as np
import nibabel as nib

from collections import OrderedDict as odict
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import zoom
import imageio
import matplotlib.pyplot as plt
from sklearn import metrics


class SimpleTimer(object):
    def __init__(self):
        self.start_time = None
        self.expire = 0
        self.counter = 0

    def tic(self):
        self.start_time = time.time()
        self.counter += 1

    def toc(self):
        expire = time.time() - self.start_time
        self.expire += expire
        return expire

    def stat(self):
        return self.expire/self.counter

    def total(self):
        return (int(self.expire//3600),
                int(self.expire % 3600//60),
                int(self.expire % 3600 % 60))


class ScoreReport(object):
    def __init__(self):
        self.y_true = []
        self.y_score = []
        self.runnig_loss = 0
        self._loss_count = 0

    def clear(self):
        self.__init__()

    def _detach_cpu(self, data):
        if  data.requires_grad:
            data=data.detach()
        if data.is_cuda:
            data = data.cpu()
        return data

    def update_true(self, data):
        data = self._detach_cpu(data)
        self.y_true += data.tolist()

    def update_score(self, data):
        data = self._detach_cpu(data)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        self.y_score += data.tolist()

    def update_loss(self, data):
        data = self._detach_cpu(data)
        self.runnig_loss += data
        self._loss_count += 1

    @property
    def y_pred(self):
        return [np.argmax(score) for score in self.y_score]
    @property
    def accuracy(self):
        #print(len(self.y_true), len(self.y_pred))
        return metrics.accuracy_score(self.y_true, self.y_pred)
    @property
    def loss(self):
        return self.runnig_loss / self._loss_count


def print_model_parameters(model, verbose=False):
    total = odict()
    for name, parameter in model.named_parameters():
        num_params = np.prod(parameter.shape)
        if verbose:
            print(name, end=' ')
            print(parameter.shape, num_params)

        try:
            total[name.split('.')[0]] += num_params
        except KeyError:
            total[name.split('.')[0]] = num_params

    print()
    for key in total.keys():
        print(key, total[key])
    print('total: ', sum(total.values()))


def copy_mmap(mmap):
    t = np.empty(mmap.shape, dtype=mmap.dtype)
    t[:] = mmap[:]
    return t


def list_merge(l):
    mgd = []
    for item in l:
        assert isinstance(item, list)
        mgd += item
    return mgd


def _extract_descrip(sample):
    descrip = sample.header['descrip'][()].decode('utf-8').split(';')
    return descrip


def save_checkpoint(FG, name, state, is_best):
    root = FG.checkpoint_root
    running_fold = FG.running_fold
    prefix = name

    path = os.path.join(root, prefix)
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, 'k'+str(running_fold))
    if not os.path.exists(path):
        os.mkdir(path)

    filename = str(state['epoch'])+'.pth.tar'
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(os.path.join(path, filename)),
                        os.path.join(path, 'best.pth.tar'))


def load_checkpoint(root, running_fold, name, model, epoch, is_best, optimizer=None, device=None, verbose=True):
    prefix = name
    path = os.path.join(root, prefix, 'k'+str(running_fold))
    best_filepath = os.path.join(path, 'best.pth.tar')
    checkpoints = glob(os.path.join(path, '*.pth.tar'))
    if is_best:
        if best_filepath in checkpoints:
            filepath = best_filepath
        else:
            raise FileNotFoundError(best_filepath)
    else:
        if best_filepath in checkpoints:
            checkpoints.remove(best_filepath)
        if epoch == -1:
            checkpoints = sorted(
                checkpoints,
                key=lambda x: int(os.path.split(x)[-1].split('.')[0]))
            filepath = checkpoints[-1]
        else:
            filepath = os.path.join(path, str(epoch)+'.pth.tar')
            if os.path.isfile(filepath):
                pass
            else:
                raise FileNotFoundError(filepath)
    if device is not None:
        state = torch.load(filepath, lambda s, l: s.cuda(device))
    else:
        state = torch.load(filepath, lambda s, l: s)
    epoch = state['epoch']
    best_score = state['best_score']
    state_dict = state['state_dict']
    opt_state_dict = state['optimizer_state_dict']
    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(opt_state_dict)
    if verbose:
        print('load from', filepath)
    return epoch, best_score

def pickle_load(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

###########################################################################
def save_images(images, size, image_path, axis):
    image = np.array(merge(images, size, axis))
    return imageio.imwrite(image_path, image)


def merge(images, size, axis):
    w, h, d = images.shape[1], images.shape[2], images.shape[3]
    img=[]
    c=images.shape[4]
    if axis == 'wh':
        img = np.zeros((w*size[0], h*size[1],c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            ci = (np.array(image.shape)/2).astype(int)
            #print(ci[0])
            img[j*w:j*w+w, i*h:i*h+h, :] = image[:,:,ci[2]]

    elif axis == 'hd':
        img = np.zeros((h*size[1], d*size[2],c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            ci = (np.array(image.shape)/2).astype(int)
            img[j*h : j*h+h, i*d : i*d+d,:] = image[ci[0],:,:]

    elif axis == 'dw':
        img = np.zeros((w*size[0], d*size[2], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            ci = (np.array(image.shape)/2).astype(int)
            img[j*w : j*w+w, i*d : i*d+d,:] = image[:,ci[1],:]
    return np.squeeze(img)



def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)


def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['info_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.plot(x, y3, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)

# class Summary(object):
#     def __init__(self, **kwarg):
#         self.viz = visdom.Visdom(**kwarg)
#         self.windows = dict()
#
#     def _to_numpy(self, image):
#         if isinstance(image, torch.autograd.Variable):
#             image = image.data
#         if isinstance(image, np.ndarray):
#             pass
#         else:
#             if image.is_cuda:
#                 image = image.cpu()
#             if isinstance(image, torch.HalfTensor):
#                 image = image.float()
#             image = image.numpy()
#
#         if image.dtype != np.float32 or image.dtype != float:
#             image = image.astype(float)
#
#         return image
#
#     def _pad_to(self, images, shape):
#         new_one = []
#         for image in images:
#             ish = image.shape
#             image = np.concatenate([
#                 np.ones((shape[0]-ish[0], ish[1])),
#                 image
#             ], axis=0)
#             ish = image.shape
#             image = np.concatenate([
#                 np.ones((ish[0], shape[1]-ish[1])),
#                 image
#             ], axis=1)
#             new_one += [image[np.newaxis]]
#         return np.vstack(new_one)[np.newaxis].transpose(1, 0, 2, 3)
#
#     def image3d(self, name, image, mean=0, std=1):
#         image = self._to_numpy(image)
#         if image.ndim == 3:
#             image = image[np.newaxis]
#         elif image.ndim == 4:
#             assert image.shape[1] == 1, 'Not supported channel {}'.format(
#                 image.shape)
#         elif image.ndim == 5:
#             image = image[0, ...]
#         else:
#             raise ValueError('Not supported ndim {}'.format(image.ndim))
#
#         image = (image * std)+mean
#         image /= image.max()
#
#         ci = (np.array(image.shape)/2).astype(int)
#         collection = []
#         for i in range(image.shape[0]):
#             center_images = [image[i, ci[1], :, :].squeeze(),
#                              image[i, :, ci[2], :].squeeze(),
#                              image[i, :, :, ci[3]].squeeze()]
#             collection += [center_images]
#
#         center_image_shape = np.array(center_images[0].shape)
#         for i in range(1, 3):
#             center_image_shape = np.vstack([center_image_shape,
#                                             center_images[i].shape])
#
#         center_image_shape = np.max(center_image_shape, axis=0)
#         images = list(map(lambda i: self._pad_to(collection[i],
#                                                  center_image_shape),
#                           list(range(image.shape[0]))))
#         images = np.concatenate(images)
#         self.viz.images(images, win=name, nrow=3,
#                         opts=dict(caption=name, title=name))
#
#     def image3d_ex(self, name, image, num_images=5):
#         image = self._to_numpy(image)
#         if image.ndim == 3:
#             pass
#         elif image.ndim == 4:
#             assert image.shape[1] == 1, 'Not supported channel'
#         elif image.ndim == 5:
#             image = image[0, ...].squeeze()
#         else:
#             raise ValueError('Not supported ndim')
#
#         image += np.abs(image.min())
#         image /= image.max()
#
#         getters = []
#         for axis in image.shape:
#             getters += [np.linspace(0, axis-1, num_images, dtype=int)]
#         vizimages = [image[getters[0], :, :][np.newaxis].transpose(1, 0, 2, 3)]
#         vizimages += [image[:, getters[1], :]
#                       [np.newaxis].transpose(2, 0, 1, 3)]
#         vizimages += [image[:, :, getters[2]]
#                       [np.newaxis].transpose(3, 0, 1, 2)]
#
#         for i, vizimage in enumerate(vizimages):
#             self.viz.images(vizimage, win=name+str(i), nrow=5,
#                             opts=dict(caption=name+str(i), title=name+str(i)))
#
#     def _scalar_rectifier(self, x, y):
#         # handle data type
#         # handle torch tensors
#         x = x.data if isinstance(x, torch.autograd.Variable) else x
#         y = y.data if isinstance(y, torch.autograd.Variable) else y
#         try:
#             y = y.cpu().numpy() if y.is_cuda else y.numpy()
#         except AttributeError:
#             pass
#         try:
#             x = x.cpu().numpy() if x.is_cuda else x.numpy()
#         except AttributeError:
#             pass
#
#         # handle unsupported data type
#         if type(x) not in (int, float, np.int_, np.float_, np.ndarray):
#             raise ValueError('Cannot handle x data type {}'.format(type(x)))
#         elif type(x) is np.ndarray:
#             assert x.ndim == 1, 'x ndim > 1 is not supported'
#         if type(y) not in (int, float, np.int_, np.float_, np.ndarray):
#             raise ValueError('Cannot handle y data type {}'.format(type(y)))
#         elif type(y) is np.ndarray:
#             assert y.ndim == 1, 'y ndim > 1 is not supported'
#
#         # convert to numpy array
#         x = np.array([x]) if type(x) in (int, np.int_, float, np.float_) else x
#         y = np.array([y]) if type(y) in (int, np.int_, float, np.float_) else y
#
#         return x, y
#
#     def scalar(self, win, name, x, y, **kwarg):
#         """ Constantly update plot with update_interval
#         Input:
#             win:    str, the name of window
#             name:   str list, the name table of values this will be legend
#             x,y:    one dimensional scalar list or ndarray or Variable or
#                         torch Tensor, x,y values w.r.t. name table
#         """
#         x, y = self._scalar_rectifier(x, y)
#
#         try:
#             exists = self.windows[win]
#         except KeyError:
#             self.windows[win] = self.viz.win_exists(win)
#             exists = self.windows[win]
#
#         if exists:
#             self.viz.updateTrace(X=x, Y=y, win=win, name=name, append=True,
#                                  opts=kwarg)
#         else:
#             self.viz.line(X=x, Y=y, win=win,
#                           opts=dict(legend=[name], showlegend=True,
#                                     title=win, **kwarg))
#             self.windows[win] = True
#
#     def cam3d(self, name, image, feature, weights, target, num_images=5):
#         feature = feature.squeeze()
#
#         c, x, y, z = feature.shape
#         cam = weights[target].dot(feature.reshape(c, x*y*z))
#         cam = cam.reshape(x, y, z)
#         cam -= cam.min()
#         cam = (cam/cam.max()*255).astype(np.uint8)
#         cam = zoom(cam, np.array(image.squeeze().shape)/np.array([x, y, z]))
#         # cam[cam < 150] = 0
#         cmap = plt.get_cmap('jet')
#         cam = np.delete(cmap(cam), 3, 3).transpose(3, 0, 1, 2)
#
#         image = self._to_numpy(image)
#         image = rescale_intensity(image, in_range=(image.min(), image.max()),
#                                                    out_range=(0,255))
#         image = np.concatenate([image, image, image])
#         #image = image.squeeze()
#
#         #scaled = image+(cams*255)/100
#         scaled = image*(cams > 20)
#         scaled = rescale_intensity(scaled, in_range=(scaled.min(), scaled.max()),
#                                    out_range=(0, 255))
#         get = []
#         for axis in image.shape:
#             get += [np.linspace(0, axis-1, num_images, dtype=int)]
#
#         vizimage = [scaled[:, get[1],:,:].transpose(1,0,2,3)]
#         vizimage += [scaled[:,:,get[2],:].transpose(2,0,1,3)]
#         vizimage += [scaled[:,:,:,get[3]].transpose(3,0,1,2)]
#
#         for i, im in enumerate(vizimage):
#             self.viz.images(im, win=name+'_'+str(i), nrow=5,
#                             opts=dict(caption=name+'_'+str(i),
#                                           title=name+'_'+str(i)))
#
#     def precision_recall_curve(self, y_true, y_score, true_label, name):
#         precision, recall, _ = metrics.precision_recall_curve(
#             y_true, [score[true_label] for score in y_score],
#             true_label)
#         ap = metrics.auc(recall, precision)
#
#         if self.viz.win_exists('prcurv'+str(true_label)):
#             self.viz.updateTrace(Y=precision, X=recall,
#                                  win='prcurv'+str(true_label),
#                                  name=name+' AP: {:.3f}'.format(ap),
#                                  append=False)
#         else:
#             self.viz.line(Y=precision, X=recall, win='prcurv'+str(true_label),
#                           opts=dict(legend=[name+' AP: {:.3f}'.format(ap)],
#                                     title='Precision Recall Curve ' +
#                                     str(true_label),
#                                     xlabel='recall',
#                                     ylabel='precision'))
#
#     def roc_curve(self, y_true, y_score, true_label, name):
#         fpr, tpr, _ = metrics.roc_curve(
#             y_true, [score[true_label] for score in y_score],
#             true_label)
#         auc = metrics.auc(fpr, tpr)
#
#         if self.viz.win_exists('roccurv'+str(true_label)):
#             self.viz.updateTrace(Y=tpr, X=fpr,
#                                  win='roccurv'+str(true_label),
#                                  name=name+' AUC: {:.3f}'.format(auc),
#                                  append=False)
#         else:
#             self.viz.line(Y=tpr, X=fpr, win='roccurv'+str(true_label),
#                           opts=dict(legend=[name+' AUC: {:.3f}'.format(auc)],
#                                     title='ROC Curve',
#                                     xlabel='False Positive Rate',
#                                     ylabel='True Positive Rate'))
#
#     def close(self, win):
#         self.viz.close(win, self.viz.env)
#

if __name__ == '__main__':
    viz = visdom.Visdom(port=10001, env='temp')

    smy = Summary(visdom.Visdom(port=10001, env='temp'))

    for i in range(1000):
        smy.image3d('fdsa', np.random.rand(4, 16, 25, 14, 53))
        exit()
