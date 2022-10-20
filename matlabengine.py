import os
from functools import partial

import matlab.engine
import numpy as np

m2np_types = (matlab.double, matlab.single, matlab.int64,
              matlab.int32, matlab.int16, matlab.int8,
              matlab.uint64, matlab.uint32, matlab.uint16,
              matlab.uint8)


def m2np(x):
    if isinstance(x, m2np_types):
        return np.asarray(x._data).reshape(x.size, order='F')
    else:
        return x


class MatRet(object):
    def __init__(self, result, nargout=None):
        self._ret = result
        self._nargout = nargout

    def __iter__(self):
        if self._nargout:
            for v in self.result:
                yield m2np(v)
        else:
            yield m2np(self.result)

    @property
    def result(self):
        assert isinstance(self._ret, matlab.engine.FutureResult)
        return self._ret.result()


class MatEng(object):
    def __init__(self, **kwargs):
        self._eng = matlab.engine.start_matlab(background=True, **kwargs)

    @property
    def m_eng(self):
        if isinstance(self._eng, matlab.engine.FutureResult):
            source_path = os.path.dirname(os.path.abspath(__file__))
            gen_path = [f.path for f in os.scandir(source_path) if f.is_dir() and not f.name.startswith('.')]
            self._eng = self._eng.result()
            self._eng.addpath(*gen_path)

        return self._eng

    def __getattr__(self, item):
        func = partial(getattr(self.m_eng, item), background=True)
        return lambda *args, **kwargs: MatRet(func(*args, **kwargs), kwargs.get('nargout'))
