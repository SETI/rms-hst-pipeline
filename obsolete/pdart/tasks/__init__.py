"""The pdart.task package contains code to queue tasks up to run them in
parallel.

One of the goals of the PDART project is to create a software pipeline
to download and process Hubble imagery.  The pipeline may not run 24/7
and so it's possible that a given task might outlive the execution of
the program.  We need some way to capture "the rest of the
calculation" (sometimes called the *continuation* in computer science)
and save it, so the pipeline's work can be picked up and resumed in
the next run of the pipeline software.

We use a simple method to implement this.  Work to do is encoded as a
:class:`~pdart.task.Task.Task`.  When a processor is free to run it,
its :meth:`~pdart.task.Task.Task.run` method is called.  After it
completes, its :meth:`~pdart.task.Task.Task.on_success` hook is called
to queue up any follow-up tasks.

In this way, instead of coding up one long function that does ``A``,
then ``B``, then ``C``, then ``D``, you code an ``ATask`` that does
``A`` when run and then queues up a ``BTask`` when it finishes.  The
``BTask`` does ``B`` and queues up a ``CTask`` when it finishes, and
so on.

If the pipeline is stopped at any point, the queue of pending tasks is
written to disk, and will be re-read when the pipeline is re-started.
So the order in which tasks need to be done, rather than being encoded
into a single function, gets encoded into a chain of tasks, each task
queueing up its successor.  The existence of a task in the queue lets
you know where you are in the process.

Other hook methods exist for other cases.  There is also an
:meth:`~pdart.task.Task.Task.on_failure` hook that is called if the
task fails.  :meth:`~pdart.task.Task.Task.on_timeout` is called on
tasks that take longer than expected and get interrupted, and
:meth:`~pdart.task.Task.Task.on_termination` is called for tasks that
are forceably ended when the pipeline shuts down.

**New to PDART?** :mod:`~pdart.task.Task` contains the base class for
tasks to be run.  :mod:`~pdart.task.NullTask` contains utility tasks
for testing.

Tasks run in separate system processes, represented by
:class:`~pdart.task.TaskProcess.TaskProcess`.  The module
:mod:`~pdart.task.RawProcess` contains a chain of classes building up
the underlying implementation for
:class:`~pdart.task.TaskProcess.TaskProcess` and can be ignored.

The tasks that are to be run and that are running are kept in a
:class:`~pdart.task.TaskQueue.TaskQueue`.  The tasks currently running
are stored in a :class:`~pdart.task.TaskDict.TaskDict` inside the
:class:`~pdart.task.TaskQueue.TaskQueue`; it maps
:class:`~pdart.task.Task.Task` s to the
:class:`~pdart.task.TaskProcess.TaskProcess` es they're running in.
The :class:`~pdart.task.TaskRunner.TaskRunner` contains the queue and
manages it: this is the top-level class and you can probably ignore
the other classes, which make up its implementation.

"""
