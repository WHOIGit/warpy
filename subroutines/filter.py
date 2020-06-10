import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def maskc(maska, pts, arr):

    pts_A = np.array(pts)

    maskx, masky = np.shape(maska)

    min_x = int(np.min(pts_A[:, 0]))
    max_x = int(np.max(pts_A[:, 0]))

    min_y = int(np.min(pts_A[:, 1]))
    max_y = int(np.max(pts_A[:, 1]))

    diff_x = max_x - min_x
    diff_y = max_y - min_y

    if diff_x < maskx:
        max_x = max_x + 1

    if diff_y < masky:
        max_y = max_y + 1


    final_mask = np.zeros_like((arr))

    final_mask[min_x:max_x,min_y:max_y] = maska

    return final_mask

def pol(arr):
    ## Create image to display
    # arr = np.random.random((100,100))

    ## create GUI
    app = QtGui.QApplication([])
    w = pg.GraphicsWindow(size=(1000, 800), border=True)
    w.setWindowTitle('pyqtgraph example: ROI Examples')

    text = """Data Selection From Image.<br>\n
    Drag an ROI """

    w1 = w.addLayout(row=0, col=0)
    label1 = w1.addLabel(text, row=0, col=0)
    v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
    img1a = pg.ImageItem(arr)

    # Get the colormap
    colormap = cm.get_cmap("viridis")  # cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

    # Apply the colormap
    img1a.setLookupTable(lut)

    v1a.addItem(img1a)

    v1a.disableAutoRange('xy')

    v1a.autoRange()

    rois = []

    rois.append(pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6, 9), closed=True))

    def update(roi):
        roi.getArrayRegion(arr, img1a)

    for roi in rois:
        roi.sigRegionChanged.connect(update)
        v1a.addItem(roi)

    # update(rois[-1])

    # mask=roi.getArrayRegion(arr, img1a)

    ## Start Qt event loop unless running in interactive mode or using pyside.
    if __name__ == '__main__':
        import sys

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    mask = roi.getArrayRegion(arr, img1a)
    pts = roi.getState()['points']

    return mask, pts