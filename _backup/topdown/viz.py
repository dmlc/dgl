
import visdom
import matplotlib.pyplot as PL
from util import *
import numpy as np
import cv2

def _fig_to_ndarray(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    data = data.transpose(2, 0, 1)
    PL.close(fig)

    return data

class VisdomWindowManager(visdom.Visdom):
    def __init__(self, **kwargs):
        visdom.Visdom.__init__(self, **kwargs)
        self.scalar_plot_length = {}
        self.scalar_plot_prev_point = {}

        self.mpl_figure_sequence = {}

    def append_scalar(self, name, value, t=None, opts=None):
        if self.scalar_plot_length.get(name, 0) == 0:
            # If we are creating a scalar plot, store the starting point but
            # don't plot anything yet
            self.close(name)
            t = 0 if t is None else t
            self.scalar_plot_length[name] = 0
        else:
            # If we have at least two values, then plot a segment
            t = self.scalar_plot_length[name] if t is None else t
            prev_v, prev_t = self.scalar_plot_prev_point[name]
            newopts = {'xlabel': 'time', 'ylabel': name}
            if opts is not None:
                newopts.update(opts)
            self.line(
                    X=np.array([prev_t, t]),
                    Y=np.array([prev_v, value]),
                    win=name,
                    update=None if not self.win_exists(name) else 'append',
                    opts=newopts
                    )

        self.scalar_plot_prev_point[name] = (value, t)
        self.scalar_plot_length[name] += 1

    def display_mpl_figure(self, fig, **kwargs):
        '''
        Call this function before calling 'PL.show()' or 'PL.savefig()'.
        '''
        self.image(
                _fig_to_ndarray(fig),
                **kwargs
                )

    def reset_mpl_figure_sequence(self, name):
        self.mpl_figure_sequence[name] = []

    def append_mpl_figure_to_sequence(self, name, fig):
        data = _fig_to_ndarray(fig)
        data = data.transpose(1, 2, 0)
        if name not in self.mpl_figure_sequence:
            self.reset_mpl_figure_sequence(name)
        self.mpl_figure_sequence[name].append(data)

    def display_mpl_figure_sequence(self, name, **kwargs):
        data_seq = self.mpl_figure_sequence[name]
        video_rows, video_cols = data_seq[0].shape[:2]
        data_seq = [cv2.resize(f, (video_cols, video_rows)) for f in data_seq]
        data_seq = np.array(data_seq, dtype=np.uint8)

        self.video(
                data_seq,
                **kwargs
                )
