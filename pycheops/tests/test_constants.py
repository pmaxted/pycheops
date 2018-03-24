
from math import pi, sqrt

from unittest import TestCase

from cheops.constants import *


class TestConstants(TestCase):

    def test_pc(self):
        assert abs(pc - 180*3600*au/pi)/pc < 1e-12


