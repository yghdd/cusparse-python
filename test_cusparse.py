import sys
import unittest

import numpy as np

import scipy
import scipy.io

import pycuda.autoinit          # noqa
import pycuda.gpuarray as gpuarray

import cusparse


EPSILON = sys.float_info.epsilon


class MatrixVectorProductTest(unittest.TestCase):
    def test_matrix_vector_product(self):
        matrix = scipy.io.mmread('test-matrix.mtx').tocsr()
        m, n = matrix.shape
        nnz = matrix.nnz
        csrValA = gpuarray.to_gpu(matrix.data.astype(np.float64))
        csrRowPtrA = gpuarray.to_gpu(matrix.indptr)
        csrColIndA = gpuarray.to_gpu(matrix.indices)
        handle = cusparse.cusparseCreate()
        descr = cusparse.cusparseCreateMatDescr()

        ones = np.ones(matrix.shape[1], dtype=np.float64)
        x = gpuarray.to_gpu(ones)
        y = gpuarray.empty_like(x)

        op = cusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
        cusparse.cusparseDcsrmv(handle, op, m, n, nnz, 1.0, descr, csrValA,
                                csrRowPtrA, csrColIndA, x, 0.0, y)

        self.assertAlmostEqual(np.sum(y).get(), 1e3, delta=EPSILON)

        cusparse.cusparseDestroyMatDescr(descr)
        cusparse.cusparseDestroy(handle)


if __name__ == '__main__':
    unittest.main()
