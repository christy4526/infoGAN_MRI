from __future__ import print_function, division, absolute_import
import torch
import numpy as np


class Scalar(object):
    def __init__(self, vis, win, opts):
        self.vis = vis
        self.win = win
        self.removed = False
        self.opts = opts

    def _rectify(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().item()
        return np.array([x])

    def __call__(self, name, x, y, remove=False):
        x, y = self._rectify(x), self._rectify(y)
        if remove:
            res = self.vis.line(X=x, Y=y, win=self.win, name=name, update='remove',
                                opts=self.opts)
            self.removed = True
        else:
            res = self.vis.line(X=x, Y=y, win=self.win, name=name, update='append',
                                opts=self.opts)
        if res == 'win does not exist':
            self.vis.line(X=x, Y=y, win=self.win, name=name,
                          opts=self.opts)


class Image3D(object):
    def __init__(self, vis, win):
        self.vis = vis
        self.win = win

    def _to_numpy(self, x):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().numpy()
        x = x.astype(float)
        if x.ndim == 5:
            x = x[0, ...]
            assert x.shape[0] == 1, 'multi-channel not supported'
            x = x.squeeze()
        elif x.ndim == 4:
            raise AssertionError('4-D image not supported')
        elif x.ndim == 3:
            pass
        else:
            raise AssertionError('{}-D image not supported'.format(
                x.ndim))
        return x

    def _same_size(self, images):
        shapes = [image.shape for image in images]
        pannel_shape = np.max(shapes, axis=0)
        pannels = [np.zeros(pannel_shape) for _ in range(len(images))]
        for pannel, image in zip(pannels, images):
            s = image.shape
            pannel[:s[0], :s[1]] += image
        return pannels

    def _slice_stack(self, image, nim):
        ci = (np.array(image.shape)/2).astype(int)
        if nim == 1:
            images = [image[ci[0], :, :],
                      image[:, ci[1], :],
                      image[:, :, ci[2]]]
        else:
            getters = [np.linspace(0, axis-1, nim, dtype=int).tolist()
                       for axis in image.shape]
            images = []
            for i0, i1, i2 in zip(*getters):
                images += [image[i0, :, :]]
                images += [image[:, i1, :]]
                images += [image[:, :, i2]]
        images = self._same_size(images)
        images = np.concatenate([image[np.newaxis] for image in images])
        images = images[np.newaxis].transpose(1, 0, 2, 3)
        return images

    def __call__(self, name, image, nimages=1):
        if not isinstance(image, np.ndarray):
            image = self._to_numpy(image)
        assert image.ndim == 3
        images = self._slice_stack(image, nimages)
        self.vis.images(images, win=self.win, nrow=3,
                        opts=dict(caption=name, title=name))
