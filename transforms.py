from __future__ import absolute_import, division, print_function

import torch
import torchvision
import random
import numpy as np


class ToWoldCoordinateSystem(object):
    def __call__(self, data):
        data = np.rot90(data.T, 2)
        return data.copy()


class Crop(object):
    def __init__(self, *cropping):
        c = cropping
        self.cropping = (slice(c[2], -c[3] if c[3] != 0 else None),
                         slice(c[4], -c[5] if c[5] != 0 else None),
                         slice(c[0], -c[1] if c[1] != 0 else None))

    def __call__(self, data):
        data = data[self.cropping[0], self.cropping[1], self.cropping[2]]
        return data


class RandomCrop(object):
    def __init__(self, *cropping):
        self.cropping = cropping

    def __call__(self, data):
        c = [random.choice(list(range(0, data.shape[i]-self.cropping[i]-1)))
             for i in range(len(self.cropping))]
        # print(range(c[0],c[0]+self.cropping[0]))
        # print(range(c[1],c[1]+self.cropping[1]))
        # print(range(c[2],c[2]+self.cropping[2]))
        return data[c[0]:c[0]+self.cropping[0],
                    c[1]:c[1]+self.cropping[1],
                    c[2]:c[2]+self.cropping[2]]
    """
    def __init__(self, cropping):
        self.cropping = cropping

    def _crop(self, data, slices):
        return data[slices[0], slices[1], slices[2]]

    def __call__(self, mri):
        mri_idx = [random.choice(list(range(0, mri.shape[i]-self.cropping[i]-1)))
                   for i in range(len(self.cropping))]
        mri_slices = [slice(mri_idx[0], mri_idx[0]+self.cropping[0])
                      for i in range(len(self.cropping))]
        return self._crop(mri, mri_slices)
    """

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if (np.array(image.shape) == np.array(self.size)).all():
            return image

        s = np.array(self.size)
        r = np.array(image.shape) - s
        m = (r/2).astype(int)
        mr = np.array(image.shape)-s-m

        return image[m[0]:image.shape[0]-mr[0],
                     m[1]:image.shape[1]-mr[1],
                     m[2]:image.shape[2]-mr[2]]


class Pad(object):
    def __init__(self, *padding):
        self.padding = padding

    def __call__(self, data):
        return torch.nn.functional.pad(data, self.padding)


class ToTensor(object):
    def __call__(self, data):
        data = torch.from_numpy(data[np.newaxis]).float()
        data /= 255
        return data


class ToFloatTensor(object):
    def __call__(self, data):
        data = data.float()
        data /= 255
        return data


class Normalize(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = (data-self.mean)/self.std
        return data


class Flip(object):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, image):
        return np.flip(image, self.axis)


class AllFlip(object):
    def __call__(self, image):
        images = []
        for axis in range(3):
            images += [np.flip(image, axis).copy()]
        return images


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if (np.array(image.shape) == np.array(self.size)).all():
            return image

        s = np.array(self.size)
        r = np.array(image.shape) - s
        m = (r/2).astype(int)
        mr = np.array(image.shape)-s-m

        return image[m[0]:image.shape[0]-mr[0],
                     m[1]:image.shape[1]-mr[1],
                     m[2]:image.shape[2]-mr[2]]


class PartialNineCrop(object):
    def __init__(self, inshape, outshape, index):
        self.inshape = np.array(inshape)
        self.outshape = np.array(outshape)

        ins = self.inshape
        ots = self.outshape
        rem = np.array(ins) - np.array(ots)
        m = (rem/2).astype(int)
        mr = ots + m
        self.index = index
        self.indices = [
            (slice(0, ots[0]),    slice(0, ots[1]),    slice(0, ots[2])),
            (slice(0, ots[0]),    slice(rem[1], None), slice(0, ots[2])),
            (slice(0, ots[0]),    slice(0, ots[1]),    slice(rem[2], None)),
            (slice(0, ots[0]),    slice(rem[1], None), slice(rem[2], None)),

            (slice(rem[0], None), slice(0, ots[1]),    slice(0, ots[2])),
            (slice(rem[0], None), slice(rem[1], None), slice(0, ots[2])),
            (slice(rem[0], None), slice(0, ots[1]),    slice(rem[2], None)),
            (slice(rem[0], None), slice(rem[1], None), slice(rem[2], None)),

            (slice(m[0], mr[0]), slice(m[1], mr[1]), slice(m[2], mr[2]))
        ]

    def __call__(self, image):
        return image[self.indices[self.index]]


class NineCrop(PartialNineCrop):
    def __init__(self, inshape, outshape):
        super(NineCrop, self).__init__(inshape, outshape, None)

    def __call__(self, image):
        patches = []
        for i in range(9):
            self.index = i
            patches += [super(NineCrop, self).__call__(image)]
        return patches


class NineCropFlip(NineCrop):
    # get each corner and center cropped image patchs and their flip(all axis)
    def __init__(self, size, axis):
        raise NotImplementedError
        super(NineCropFlip, self).__init__(size)
        self.flip = Flip(axis)

    def __call__(self, image):
        images = super(NineCropFlip, self).__call__(image)

        flipped = []
        for cim in images:
            flipped += [self.flip(cim)]
        return images+flipped


class RandomFlip(object):
    def __init__(self, axis, p=0.5):
        self.axis = axis
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return np.flip(image, self.axis)
        return image


class RandomInverse(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return image.max() - image
        return image


class Resize(object):
    def __init__(self, shape):
        self.shape = np.array(shape)

    def __call__(self, image):
        im_shape = np.array(image.shape)
        return zoom(image, self.shape/im_shape)


class Stack(object):
    def __call__(self, images):
        return torch.stack(images)



if __name__ == '__main__':
    data = torch.rand(79, 95, 68)
    data = Crop(2, 2, 0, 0, 0, 0)(data)
    data = Pad(0, 0, 0, 1, 0, 1)(data)
    print(data.shape)
