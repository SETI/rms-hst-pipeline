import unittest

from pdart.rules.Combinators import *


class TestCombinators(unittest.TestCase):

    def test_normalized_exceptions(self):
        # type: () -> None
        def explode():
            raise Exception('Boom!')
        explode = normalized_exceptions(explode)
        try:
            explode()
            self.assertTrue(False)
        except CalculationException as e:
            pass  # expected

    def test_multiple_implementations(self):
        # type: () -> None
        def failure_func(n):
            def func():
                """Fail."""
                raise Exception('Error %d' % n)
            func.__name__ = 'failure_func(%d)' % n
            return func

        def success_func():
            """Succeed."""
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

        expected_name = 'multiple_implementations(' + \
            "'test_multiple_implementations', " + \
            'failure_func(1), ' + \
            'failure_func(2), ' + \
            'success_func)'
        self.assertEquals(expected_name, mi.__name__)

        self.assertTrue('failure_func' in mi.__doc__)
        self.assertTrue('Fail.' in mi.__doc__)
        self.assertTrue('success_func' in mi.__doc__)
        self.assertTrue('Succeed.' in mi.__doc__)

    def test_parallel_list(self):
        # type: () -> None
        # TODO
        pass
