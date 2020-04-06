#!/usr/bin/env python

import sys
import os
import time
import argparse
import configparser

import h5py
import numpy as np
import cupy as cp

class DataGenerator():
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.num_data = config.getint('make_data', 'num_data')
        self.fluence = config.get('make_data', 'fluence', fallback='constant')
        self.mean_count = config.getfloat('make_data', 'mean_count')
        self.bg_count = config.getfloat('make_data', 'bg_count', fallback=None)
        self.rel_scale = config.getfloat('make_data', 'rel_scale', fallback=1000.)
        self.dia_params = [float(s) for s in config.get('make_data', 'dia_params').split()]
        self.shift_sigma = config.getfloat('make_data', 'shift_sigma')
        self.out_file = os.path.join(os.path.dirname(config_file),
                                     config.get('make_data', 'out_photons_file'))

        if self.fluence not in ['constant', 'gamma']:
            raise ValueError('make_data:fluence needs to be either constant (default) or gamma')
        with open('kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen_holo = kernels.get_function('slice_gen_holo')
        self.k_slice_gen = kernels.get_function('slice_gen')
        self.object = cp.zeros((self.size, self.size), dtype='f8')
        self.object_sum = 0
        self.bgmask = cp.zeros_like(self.object)
        self.bgmask_sum = 0

    def make_obj(self, bg=False):
        mask = self.bgmask if bg else self.object

        num_circ = 80
        mcen = self.size // 2
        x, y = cp.indices(self.object.shape, dtype='f8')
        for _ in range(num_circ):
            rad = (0.7 + 0.3*cp.random.rand(1, dtype='f8')) * self.size / 25.
            while True:
                cen = cp.random.rand(2, dtype='f8') * self.size / 5. + mcen * 4./ 5.
                dist = float(cp.sqrt((cen[0]-mcen)**2 + (cen[1]-mcen)**2) + rad)
                if dist < mcen:
                    break

            diskrad = cp.sqrt((x - cen[0])**2 + (y - cen[1])**2)
            mask[diskrad <= rad] += 1. - (diskrad[diskrad <= rad] / rad)**2

        if bg:
            #mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask.sum())
        else:
            #mask *= self.mean_count / mask.sum()
            self.object_sum = float(mask.sum())

            with h5py.File(self.out_file, 'a') as fptr:
                if 'solution' in fptr:
                    del fptr['solution']
                fptr['solution'] = mask.get()

    def parse_obj(self, bg=False):
        mask = self.bgmask if bg else self.object
        dset_name = 'bg' if bg else 'solution'

        with h5py.File(self.out_file, 'r') as fptr:
            mask = cp.array(fptr[dset_name][:])

        if bg:
            mask *= self.bg_count / mask.sum()
            self.bgmask_sum = float(mask.sum())
            self.bgmask = mask
        else:
            #mask *= self.mean_count / mask.sum()
            self.object_sum = float(mask.sum())
            self.object = mask

    def make_data(self, parse=False):
        if self.object_sum == 0.:
            if parse:
                self.parse_obj()
            else:
                self.make_obj()

        if self.bg_count is not None:
            if parse:
                self.parse_obj(bg=True)
            else:
                self.make_obj(bg=True)

        mask = cp.ones(self.object.shape, dtype='f8')
        x, y = cp.indices(self.object.shape, dtype='f8')
        cen = self.size // 2
        pixrad = cp.sqrt((x - cen)**2 + (y - cen)**2)
        mask[pixrad<4] = 0
        mask[pixrad>=cen] = 0

        fptr = h5py.File(self.out_file, 'a')
        if 'ones' in fptr: del fptr['ones']
        if 'multi' in fptr: del fptr['multi']
        if 'place_ones' in fptr: del fptr['place_ones']
        if 'place_multi' in fptr: del fptr['place_multi']
        if 'count_multi' in fptr: del fptr['count_multi']
        if 'num_pix' in fptr: del fptr['num_pix']
        if 'true_shifts' in fptr: del fptr['true_shifts']
        if 'true_diameters' in fptr: del fptr['true_diameters']
        if 'true_angles' in fptr: del fptr['true_angles']
        if 'bg' in fptr: del fptr['bg']
        if 'scale' in fptr: del fptr['scale']

        if self.bgmask_sum > 0:
            fptr['bg'] = self.bgmask.get()
        fptr['num_pix'] = np.array([self.size**2])
        dtype = h5py.special_dtype(vlen=np.dtype('i4'))
        place_ones = fptr.create_dataset('place_ones', (self.num_data,), dtype=dtype)
        place_multi = fptr.create_dataset('place_multi', (self.num_data,), dtype=dtype)
        count_multi = fptr.create_dataset('count_multi', (self.num_data,), dtype=dtype)
        ones = fptr.create_dataset('ones', (self.num_data,), dtype='i4')
        multi = fptr.create_dataset('multi', (self.num_data,), dtype='i4')

        #shifts = np.random.random((self.num_data, 2))*6 - 3
        #shifts = np.random.randn(self.num_data, 2)*1.
        #shifts = np.zeros((self.num_data, 2))
        shifts = np.random.randn(self.num_data, 2)*self.shift_sigma
        fptr['true_shifts'] = shifts
        if self.fluence == 'gamma':
            scale = np.random.gamma(2., 0.5, self.num_data)
        else:
            scale = np.ones(self.num_data, dtype='f8')
        fptr['scale'] = scale
        #diameters = np.random.randn(self.num_data)*0.5 + 7.
        #diameters = np.ones(self.num_data)*7.
        diameters = np.random.randn(self.num_data)*self.dia_params[1] + self.dia_params[0]
        fptr['true_diameters'] = diameters
        #rel_scales = diameters**3 * 1000. / 7**3
        #scale *= rel_scales/1.e3
        angles = np.random.random(self.num_data) * 2. * np.pi
        #angles = np.zeros(self.num_data)
        fptr['true_angles'] = angles

        view = cp.zeros(self.size**2, dtype='f8')
        rview = cp.zeros_like(view, dtype='f8')
        zmask = cp.zeros_like(view, dtype='f8')
        model = cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(self.object)))
        bsize_model = int(np.ceil(self.size/32.))
        stime = time.time()
        for i in range(self.num_data):
            self.k_slice_gen_holo((bsize_model,)*2, (32,)*2,
                (model, shifts[i,0], shifts[i,1], diameters[i], self.rel_scale, scale[i], self.size, zmask, 0, view))
            view *= mask.ravel()
            view *= self.mean_count / view.sum()
            self.k_slice_gen((bsize_model,)*2, (32,)*2,
                (view, angles[i], 1., self.size, self.bgmask, 0, rview))
            frame = cp.random.poisson(rview, dtype='i4')
            place_ones[i] = cp.where(frame == 1)[0].get()
            place_multi[i] = cp.where(frame > 1)[0].get()
            count_multi[i] = frame[frame > 1].get()
            ones[i] = place_ones[i].shape[0]
            multi[i] = place_multi[i].shape[0]
            sys.stderr.write('\rWritten %d/%d frames (%d)  ' % (i+1, self.num_data, int(frame.sum())))
        etime = time.time()
        sys.stderr.write('\nTime taken (make_data): %f s\n' % (etime-stime))
        fptr.close()

def main():
    parser = argparse.ArgumentParser(description='Padman data generator')
    parser.add_argument('-c', '--config_file',
                        help='Path to config file (Default: config.ini)',
                        default='config.ini')
    parser.add_argument('-m', '--mask_only',
                        help='Create mask only and not the data frames',
                        action='store_true', default=False)
    parser.add_argument('-d', '--data_only',
                        help='Generate data only. Use preexisting mask in file',
                        action='store_true', default=False)
    parser.add_argument('-D', '--device',
                        help='Device number (default: 0)', type=int, default=0)
    args = parser.parse_args()

    datagen = DataGenerator(args.config_file)
    if not args.data_only:
        datagen.make_obj()
    if not args.mask_only:
        datagen.make_data(parse=args.data_only)

if __name__ == '__main__':
    main()
