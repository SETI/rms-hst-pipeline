# coding=utf-8
import os
import tempfile
import unittest

from pdart.labels.Utils import wavelength_from_range


class Test_Utils(unittest.TestCase):
    def test_wavelength_from_range(self) -> None:
        self.assertEqual(["Ultraviolet", "Visible"], wavelength_from_range(0.01, 0.4))
        self.assertEqual(
            ["Ultraviolet", "Visible", "Near Infrared"],
            wavelength_from_range(0.390, 0.7),
        )
        self.assertEqual(
            ["Visible", "Near Infrared", "Infrared"], wavelength_from_range(0.65, 5.0)
        )
        self.assertEqual(
            ["Near Infrared", "Infrared", "Far Infrared"],
            wavelength_from_range(0.75, 300),
        )
        self.assertEqual(["Ultraviolet", "Visible"], wavelength_from_range(0.395, 0.6))
        self.assertEqual(
            ["Visible", "Near Infrared", "Infrared"], wavelength_from_range(0.5, 0.8)
        )
        self.assertEqual(
            ["Visible", "Near Infrared", "Infrared"], wavelength_from_range(0.5, 0.8)
        )
        self.assertEqual(
            ["Ultraviolet", "Visible", "Near Infrared", "Infrared", "Far Infrared"],
            wavelength_from_range(0.005, 100),
        )
        self.assertEqual(
            ["Ultraviolet", "Visible", "Near Infrared", "Infrared", "Far Infrared"],
            wavelength_from_range(0.3, 305),
        )
        self.assertEqual([], wavelength_from_range(0.005, 0.006))
        self.assertEqual([], wavelength_from_range(301, 303))
        self.assertEqual(["Ultraviolet"], wavelength_from_range(0.01, 0.2))
        self.assertEqual(["Visible"], wavelength_from_range(0.5, 0.6))
        self.assertEqual(["Near Infrared"], wavelength_from_range(0.71, 0.72))
        self.assertEqual(["Infrared", "Far Infrared"], wavelength_from_range(6, 100))

        # from (microns)  to  (microns)     name           PDS4 range
        #     0.01    -           0.400     Ultraviolet    (10 and 400 nm)
        #     0.390   -           0.700     Visible        (390 and 700 nm)
        #     0.65    -           5.0       Near Infrared  (0.65 and 5.0 micrometers)
        #     0.75    -         300         Infrared       (0.75 and 300 micrometers)
        #    30       -         300         Far Infrared   (30 and 300 micrometers)
