import GPUtil
import threading


class AbortableSleep():
    """
    A class that enables sleeping with interrupts
    see https://stackoverflow.com/questions/28478291/abortable-sleep-in-python
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._aborted = False

    def __call__(self, secs):
        with self._condition:
            self._aborted = False
            self._condition.wait(timeout=secs)
            return not self._aborted

    def abort(self):
        with self._condition:
            self._condition.notify()
            self._aborted = True


class GpuLogThread(threading.Thread):
    """
    simple thread to log gpu util to tensorboard and keep track of util spikes
    not multithreading safe
    """

    def __init__(self, gpu_ids: list, writer, seconds=10, rs=0.5):
        super(GpuLogThread, self).__init__()
        self.gpu_ids = gpu_ids
        self.seconds = seconds
        self.rs = rs
        self.writer = writer
        self.step_writer = 0
        self.step_recent = 0
        self.keep_running = True
        self.max_recent_util = 0.0
        self._abortable_sleep = AbortableSleep()
        self.daemon = True

    def __log_gpus(self):
        for i, gpu in enumerate(GPUtil.getGPUs()):
            if i in self.gpu_ids:
                # self.writer.add_scalar('gpus/%d/%s' % (gpu.id, 'memoryTotal'), gpu.memoryTotal, step)
                # self.writer.add_scalar('gpus/%d/%s' % (gpu.id, 'memoryUsed'), gpu.memoryUsed, step)
                # self.writer.add_scalar('gpus/%d/%s' % (gpu.id, 'memoryFree'), gpu.memoryFree, step)
                self.writer.add_scalar('gpus/%d/%s' % (gpu.id, 'memoryUtil'), gpu.memoryUtil, self.step_writer)
            self.writer.add_scalar('gpus/recentMaxUtil', self.max_recent_util, self.step_writer)
        self.step_writer += 1

    def __update_recent(self):
        for i, gpu in enumerate(GPUtil.getGPUs()):
            if i in self.gpu_ids:
                self.max_recent_util = max(self.max_recent_util, gpu.memoryUtil)
        self.step_recent += 1

    def wakeup(self):
        self.__update_recent()
        self._abortable_sleep.abort()

    def run(self):
        while self.keep_running:
            for i in range(int(self.seconds / self.rs)):
                self.__update_recent()
                self._abortable_sleep(self.rs)
            self.__log_gpus()

    def stop(self):
        self.keep_running = False

    def get_highest_recent(self):
        usage = self.max_recent_util
        if usage <= 0:
            return 1.0
        return usage

    def reset_recent(self):
        self.max_recent_util = 0.0
