"""DGL's device context shim."""

class Context(object):
    def __init__(self, dev, devid=-1):
        self.device = dev
        self.device_id = devid

    def __str__(self):
        return '{}:{}'.format(self.device, self.device_id)

def gpu(gpuid):
    return Context('gpu', gpuid)

def cpu():
    return Context('cpu')
