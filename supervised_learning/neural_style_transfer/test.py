import unittest
import numpy as np
<<<<<<< HEAD
import tensorflow as tf
import os
=======

>>>>>>> 3fc826ad95e8de7a30b3c5c9473a3cd5947c901f

NST = __import__('0-neural_style').NST
class TestNST(unittest.TestCase):
    def test_init(self):
        style_image = np.random.randint(256, size=(300, 300, 3), dtype='uint8')
        content_image = np.random.randint(256, size=(300, 300, 3), dtype='uint8')
        alpha = 1e4
        beta = 1

        nst = NST(style_image, content_image, alpha, beta)

        self.assertTrue(np.array_equal(nst.style_image, style_image))
        self.assertTrue(np.array_equal(nst.content_image, content_image))
        self.assertEqual(nst.alpha, alpha)
        self.assertEqual(nst.beta, beta)

if __name__ == '__main__':
    unittest.main()
