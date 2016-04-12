import abc


class Result(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_success(self):
        pass

    def is_failure(self):
        return not self.is_success()


class Failure(Result):
    def __init__(self, exception_info):
        Result.__init__(self)
        self.exception_info = exception_info

    def is_success(self):
        return False

    def __str__(self):
        return '_Failure(%s)' % (self.exception_info, )


class Success(Result):
    def __init__(self, value):
        Result.__init__(self)
        self.value = value

    def is_success(self):
        return True

    def __str__(self):
        return 'Success(%s)' % (self.value, )
