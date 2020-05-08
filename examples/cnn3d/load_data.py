from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy
from six.moves import xrange
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import tensorflow as tf


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32)
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_dims(f, gridSize, nbDim):
    print('Check dimensions ', f.name,  flush = True)
    with f as bytestream:
        headerSize = _read32(bytestream)
        magic = _read32(bytestream)
        if magic != 7919:
            raise ValueError('Invalid magic number %d in maps file: %s' %
                             (magic, f.name))
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        lays = _read32(bytestream)
        assert(rows == gridSize)
        assert(cols == gridSize)
        assert(lays == gridSize)
        chan = _read32(bytestream)
        assert(chan == nbDim)


def extract_maps(f):
    #print('Extracting', f.name,  flush = True)
    with f as bytestream:
        headerSize = _read32(bytestream)
        magic = _read32(bytestream)
        if magic != 7919:
            raise ValueError('Invalid magic number %d in maps file: %s' %
                             (magic, f.name))
        rows = _read32(bytestream)
        #print("rows "+str(rows))
        cols = _read32(bytestream)
        #print("cols "+str(cols))
        lays = _read32(bytestream)
        #print("lays "+str(lays))
        chan = _read32(bytestream)
        #print("chan "+str(chan))
        metaSize = _read32(bytestream)
        #print("metaSize "+str(metaSize))
        num_maps = _read32(bytestream)
        #print("num_maps "+str(num_maps))
        header_end = bytestream.read(headerSize - 4*8)
        if num_maps<=0 :
            return None,None
        size = int(rows) * int(cols) * int(lays) * int(chan) * int(num_maps)
        size += int(metaSize) * int(num_maps)
        try :
            buf = bytestream.read(size)
        except OverflowError :
            return None, None
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_maps, -1)
        meta = numpy.ascontiguousarray(data[:, -int(metaSize):]).view(dtype=numpy.int32)
        data = data[:,:-int(metaSize)]
        return data, meta


class DataSet(object):

    def __init__(self, maps, meta, dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32, dtypes.float16):
            raise TypeError(
                'Invalid map dtype %r, expected uint8 or float32 or float16' % dtype)

        if dtype == dtypes.float32:
            maps = maps.astype(numpy.float32)
            numpy.multiply(maps, 1.0 / 255.0, out = maps)
        if dtype == dtypes.float16:
            maps = maps.astype(numpy.float16)
            numpy.multiply(maps, 1.0 / 255.0, out = maps)

        self._maps = maps
        self._meta = meta
        self._num_res = self._maps.shape[0]

    @property
    def maps(self):
        return self._maps

    @property
    def meta(self):
        return self._meta

    @property
    def num_res(self):
        return self._num_res


def read_data_set(filename, dtype=dtypes.float32):
    try :
        with open(filename, 'rb') as f:
            maps, meta = extract_maps(f)
        if maps is None:
            return None
        train = DataSet(maps, meta, dtype=dtype)
        return train
    except ValueError:
        return None
