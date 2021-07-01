"""
Functions to visualize human poses
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def show3DposePair(realt3d, faket3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=True, pred=False):  # blue, orange
  """
  Visualize a 3d skeleton pair

  Args
  channels: 96x1 vector. The pose to plot.
  ax: matplotlib 3d axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  realt3d = np.reshape(realt3d, (16, -1))
  faket3d = np.reshape(faket3d, (16, -1))

  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  for idx, vals in enumerate([realt3d, faket3d]):
    # Make connection matrix
    for i in np.arange(len(I)):
      x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
      if idx == 0:
        ax.plot(x, z, -y, lw=2, c='k')
      #        ax.plot(x,y, z,  lw=2, c='k')

      elif idx == 1:
        ax.plot(x, z, -y, lw=2, c='r')
      #        ax.plot(x,y, z,  lw=2, c='r')

      else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 1  # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

  # Get rid of the ticks and tick labels
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #  ax.set_zticks([])
  #
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])
  #  ax.set_zticklabels([])
  #     ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
               gt=False,pred=False): # blue, orange
    """
    Visualize a 3d skeleton

    Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (16, -1) )

    I  = np.array([0,1,2,0,4,5,0,7,8,8,10,11,8,13,14]) # start points
    J  = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        if gt:
            ax.plot(x,z, -y,  lw=2, c='k')
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            ax.plot(x,z, -y,  lw=2, c='r')
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            ax.plot(x, z, -y,  lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 1 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])


    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("-y")

    # Get rid of the ticks and tick labels
    #  ax.set_xticks([])
    #  ax.set_yticks([])
    #  ax.set_zticks([])
    #
    #  ax.get_xaxis().set_ticklabels([])
    #  ax.get_yaxis().set_ticklabels([])
    #  ax.set_zticklabels([])
#     ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True):
  """
  Visualize a 2d skeleton

  Args
  channels: 64x1 vector. The pose to plot.
  ax: matplotlib axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  vals = np.reshape(channels, (-1, 2))
  # plt.plot(vals[:,0], vals[:,1], 'ro')
  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange(len(I)):
    x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
    #         print('x',x)
    #         print(y)
    ax.plot(x, -y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #
  #  # Get rid of tick labels
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])

  RADIUS = 1  # space around the subject
  xroot, yroot = vals[0, 0], vals[0, 1]
  #     ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  #     ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])

  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("-y")

  ax.set_aspect('equal')


##############################
# wrap for simple usage
##############################
def wrap_show3d_pose(vals3d):
    fig3d = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d = Axes3D(fig3d)
    show3Dpose(vals3d, ax3d)
    plt.show()


def wrap_show2d_pose(vals2d):
    ax2d = plt.axes()
    show2Dpose(vals2d, ax2d)
    plt.show()