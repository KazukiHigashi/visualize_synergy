import os
import rospy

from python_qt_binding import loadUi

from PyQt5 import QtCore, Qt
from PyQt5.QtGui import QTextDocument
from controller_manager_msgs.srv import ListControllers
from sr_robot_msgs.msg import sendupdate, joint
from std_msgs.msg import Float64
from math import radians, degrees
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class InfoViewer(QTextDocument):
    def __init__(self, *__args):
        QTextDocument.__init__(self, *__args)
