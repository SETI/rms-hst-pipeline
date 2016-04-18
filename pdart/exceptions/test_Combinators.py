import unittest

from pdart.exceptions.Combinators import *


class TestCombinators(unittest.TestCase):

    def test_normalized_exceptions(self):
        def explode():
            raise Exception('Boom!')
        explode = normalized_exceptions(explode)
        try:
            explode()
            self.assertTrue(False)
        except CalculationException as e:
            pass  # expected

    def test_multiple_implementations(self):
        def failure_func(n):
            def func():
                raise Exception('Error %d' % n)
            return func

        def success_func():
            pass
        funcs = [failure_func(1), failure_func(2), failure_func(3)]
        mi = multiple_implementations('test_multiple_implementations',
                                      *funcs)

        try:
            mi()
            self.assertTrue(False)
        except CalculationException as ce:
            # TODO check the exception
            pass

        funcs = [failure_func(1), failure_func(2), success_func]
        mi = multiple_implementations('test_multiple_implementations',
                                      *funcs)
        try:
            mi()
            # okay
        except CalculationException as ce:
            print '*********', repr(ce)
            self.assertTrue(False)

    def test_parallel_list(self):
        # TODO
        pass
