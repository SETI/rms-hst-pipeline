from pdart.tasks.TaskRunner import *
from pdart.tasks.NullTask import *


def test_TaskRunner():
    # type: () -> None
    try:
        tr = TaskRunner(-4)
        assert False, "Should have raised exception"
    except:
        pass

    tr = TaskRunner(4)
    assert tr.SLEEP_TIME == 10 * 60
    tr.run_loop()  # exits immediately

    tr = TaskRunner(4)
    tr.SLEEP_TIME = 10.0 / 1000  # wake up every 10ms instead
    tasks = [NumberedNullTask() for i in xrange(0, 50)]
    # type: List[Task]

    tr.extend_pending(tasks)
    tr.run_loop()  # exits eventually


class DivideAndConquerTask(NumberedNullTask):
    """
    A NullTask given a certain number of units.  When it finishes
    doing nothing, it creates two subunits each containing half its
    units and queues them into the TaskRunner.  Running a
    DivideAndConquerTask should spawn 2^N tasks, but should also end
    once they're all reduced to units=1.
    """
    def __init__(self, units):
        NumberedNullTask.__init__(self)
        assert type(units) == int
        self.units = units

    def __str__(self):
        return 'DivideAndConquerTask(#%d, units=%d)' % (self.serial_number,
                                                        self.units)

    def to_tuple(self):
        return (self.serial_number, self.units)

    def on_success(self, target):
        # type: (TaskRunner) -> None
        if self.units > 1:
            new_units = self.units / 2
            left_task = DivideAndConquerTask(new_units)
            right_task = DivideAndConquerTask(new_units)
            target.extend_pending([left_task, right_task])

    def on_failure(self, target):
        # type: (TaskRunner) -> None
        assert False, 'DivideAndConquer.on_failure()'

    def on_termination(self, target):
        # type: (TaskRunner) -> None
        assert False, 'DivideAndConquer.on_termination()'

    def on_timeout(self, target):
        # type: (TaskRunner) -> None
        assert False, 'DivideAndConquer.on_timeout()'


def test_DivideAndConquer():
    # type: () -> None
    """
    Spawns and runs a large number of tasks (is it N^2 / 2?) to test
    the post-completion actions of the task.
    """
    N = 999
    tr = TaskRunner(4)
    tr.SLEEP_TIME = 0.01
    t = DivideAndConquerTask(N)
    tr.extend_pending([t])
    tr.run_loop()
