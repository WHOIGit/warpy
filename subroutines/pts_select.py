import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets
import numpy as np
import cv2

    
class pts_select():

    def __init__(self,im,t,f):
        self.im = im
        self.selected_points = []
        self.fig,ax = plt.subplots(figsize=[7,5])
        self.img = ax.imshow(self.im.copy(),extent=[t[0], t[-1], f[0], f[-1]], origin='low', aspect='auto')
        #self.img = ax.pcolormesh(self.im.copy(), origin='low')
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)

    def onclick(self, event):
        self.selected_points.append((event.xdata,event.ydata))
        pts = np.array(self.selected_points, np.int32)
        return pts
        
        
    def disconnect_mpl(self,_):
        self.fig.canvas.mpl_disconnect(self.ka) 
        
  