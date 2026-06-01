#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

tests for boundary conditions code
 

"""

# required so that we can find the central model code from here!

import sys
sys.path.append("../") 

import unittest
import numpy as np
from solver.physics.boundaryConditions import BCs


class BCsTestCase(unittest.TestCase):
    
    def setUp(self):
        self.xnum = 5
        self.ynum = 5
        self.BC = BCs(self.xnum, self.ynum)

    # array size checks
    def test_BC_top_array_size(self):
        self.assertEqual(np.shape(self.BC.B_top), (self.xnum+1, 4), 
                         'incorrect default B_top size')
    def test_BC_bot_array_size(self): 
        self.assertEqual(np.shape(self.BC.B_bottom), (self.xnum+1, 4), 
                         'incorrect default B_bottom size')
    def test_BC_left_array_size(self): 
        self.assertEqual(np.shape(self.BC.B_left), (self.ynum+1, 4), 
                         'incorrect default B_left size')
    def test_BC_right_array_size(self): 
        self.assertEqual(np.shape(self.BC.B_right), (self.ynum+1, 4), 
                         'incorrect default B_right size')
        
    
    # helper function checks
    def test_set_top_free_slip(self):
        self.BC.set_top_BC("free slip")
        np.testing.assert_allclose(self.BC.B_top[:,1], 1.0)
    
    def test_set_top_no_slip(self):
        self.BC.set_top_BC("no slip")
        np.testing.assert_allclose(self.BC.B_top, 0.0)
        
    def test_set_top_prescribed_v(self):
        self.BC.set_top_BC("prescribed parallel velocity", 2.0)
        np.testing.assert_allclose(self.BC.B_top[:,0], 2.0)
        
    def test_missing_v_in_prescribed(self):
        with self.assertRaises(ValueError):
            self.BC.set_top_BC("prescribed parallel velocity")
        
    



# run the tests!
if __name__ == '__main__':
    unittest.main()