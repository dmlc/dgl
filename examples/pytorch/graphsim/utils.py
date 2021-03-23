import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import matplotlib.animation as manimation

matplotlib.use('agg')
class VideoWriter:
    def __init__(self,videopath,videoname='video0',figsize=(6,6)):
        self.figsize   = figsize
        self.videopath = videopath
        self.figure = plt.figure(figsize=self.figsize)
        w,h = self.figure.canvas.get_width_height()
        self.out = cv.VideoWriter(videopath+'/{}.avi'.format(videoname),
                                  cv.VideoWriter_fourcc(*'MJPG'),
                                  15,(w,h))
        

    def fig2data(self,fig):
        import PIL.Image as Image
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8)
        buf.shape = (w,h,3)
        buf = np.roll(buf,3,axis=2)
        image = Image.frombytes('RGB',(w,h),buf.tobytes())
        image = np.asarray(image)
        return image

    def plot(self,pos):
        pos = pos.copy()
        plt.clf()
        pos *= 10
        plotter = self.figure.add_subplot(111)
        plotter.set_xlim(0,10)
        plotter.set_ylim(0,10)
        plotter.plot(pos[:,0],pos[:,1],'b.')
        image = self.fig2data(self.figure)
        image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
        self.out.write(image)
        return image

    def close(self):
        self.out.release()

def normalize_acc(acc,acc_mean,acc_std):
    acc = (acc-acc_mean)/acc_std
    return acc

# Make video can be used to visualize test data
def make_video(xy, filename):
    os.system("rm -rf pics/*")
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig = plt.figure()
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    fig_num = len(xy)
    color = ['ro', 'bo', 'go', 'ko', 'yo', 'mo', 'co']
    with writer.saving(fig, filename, len(xy)):
        for i in range(len(xy)):
            for j in range(len(xy[0])):
                plt.plot(xy[i, j, 1], xy[i, j, 0], color[j % len(color)])
            writer.grab_frame()