import unittest

import numpy as np

from powerfit_em import _powerfit


class Test_Powerfit(unittest.TestCase):

    def test_conj_multiply(self):
        # create arrays
        shape = (100,)
        in1 = np.empty(shape, dtype=np.complex128)
        in1.real = np.random.rand(*shape)
        in1.imag = np.random.rand(*shape)
        in2 = np.empty(shape, dtype=np.complex128)
        in2.real = np.random.rand(*shape)
        in2.imag = np.random.rand(*shape)
        out = np.empty(shape, dtype=np.complex128)

        # create output
        _powerfit.conj_multiply(in1, in2, out)
        answer = in1.conj() * in2

        self.assertTrue(np.allclose(answer, out))

    def test_calc_lcc(self):
        n = 100
        gcc = np.random.rand(n)
        ave = np.random.rand(n)
        # make sure ave2 is bigger than ave**2
        ave2 = 2 * ave**2
        mask = np.ones(n, dtype=np.uint8)
        lcc = np.zeros(n, dtype=np.float64)

        _powerfit.calc_lcc(gcc, ave, ave2, mask, lcc)
        answer = gcc / np.sqrt(ave2 - ave**2)

        self.assertTrue(np.allclose(answer, lcc))

if __name__ == "__main__":
    unittest.main()
