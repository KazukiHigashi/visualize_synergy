#!/usr/bin/env python
#
# Copyright 2012 Shadow Robot Company Ltd.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation version 2 of the License.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <http://www.gnu.org/licenses/>.
#

import os
import rospkg
import rospy
import math

import csv

from xml.etree import ElementTree as ET

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi

from PyQt5.QtCore import pyqtSignal, Qt, QThread
# from QtCore import Qt, QThread
from QtWidgets import QWidget, QMessageBox

from controller_manager_msgs.srv import ListControllers
from control_msgs.msg import JointControllerState
from sr_robot_msgs.msg import JointControllerState as SrJointControllerState
from sr_robot_msgs.msg import JointMusclePositionControllerState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sr_robot_msgs.msg import sendupdate

from gui_synergy_slider.sliders import JointController, Joint, EtherCATHandSlider, PrincipalComponent
from gui_synergy_slider.sliders import EtherCATHandTrajectorySlider, EtherCATSelectionSlider, \
    EtherCATHandSynergySlider
from sr_utilities.hand_finder import HandFinder

from synergy_utility import pca_utils
from math import radians, degrees
import numpy as np


def _read_mapping(path=None):
    if path is None:
        path = rospkg.RosPack().get_path('sr_cyberglove_config') + "/mappings/GloveToHandMappings_generic"

    with open(path) as f:
        reader = csv.reader(f, delimiter=' ')
        mapping_matrix = np.array([row for row in reader])
    mapping_matrix = np.array(mapping_matrix[:, 1:], dtype='float')
    return mapping_matrix # [:20, :]


class GuiSynergySlider(Plugin):
    """
    A plugin to change the position along with a principal component axis of a synergy.
    """

    # For each controller type this defines the category of controller it belongs to
    # (position, velocity, effort, position_trajectory)
    # and the msg type of the controller state topic
    controller_state_types = {
        "sr_mechanism_controllers/SrhJointPositionController": ("position", JointControllerState),
        "sr_mechanism_controllers/SrhEffortJointController": ("effort", JointControllerState),
        "sr_mechanism_controllers/SrhJointVelocityController": ("velocity", JointControllerState),
        "sr_mechanism_controllers/SrhMixedPositionVelocityJointController": ("position", SrJointControllerState),
        "sr_mechanism_controllers/SrhMuscleJointPositionController": ("position", JointMusclePositionControllerState),
        "effort_controllers/JointEffortController": ("effort", JointControllerState),
        "effort_controllers/JointPositionController": ("position", JointControllerState),
        "position_controllers/JointTrajectoryController": ("position_trajectory", JointTrajectoryControllerState),
        "effort_controllers/JointTrajectoryController": ("position_trajectory", JointTrajectoryControllerState),
        "effort_controllers/GravityCompensatedJointTrajectoryController": ("position_trajectory",
                                                                           JointTrajectoryControllerState)}

    synergy_types = {
        #  "synergy name":(index, max_pc_number)
        0: ("grasp", 10),
        1: ("sub1", 3),
        2: ("sub2", 6),
        3: ("sub3", 6),
        4: ("sub4", 6)}

    posture_update = pyqtSignal()

    def __init__(self, context):
        super(GuiSynergySlider, self).__init__(context)
        self.setObjectName('GuiSynergySlider')

        self._robot_description_xml_root = None

        self._widget = QWidget()

        ui_file = os.path.join(rospkg.RosPack().get_path(
            'gui_synergy_slider'), 'uis', 'SynergySlider.ui')
        loadUi(ui_file, self._widget)

        self._widget.setObjectName('SynergySliderUi')
        context.add_widget(self._widget)

        self.joints = []

        self.sliders = []
        self.selection_slider = None

        self.synergy = []
        self.current_synergy_type = None
        self.current_pc_num = 1
        self.current_pca_object = None
        self.selected_joints = None
        self.current_synergy_scores = []
        self.current_sr_joints_target = []

        self.joint_position_controller_pub = dict()

        # to be used for calculating next trajectory reconstructed by the synergy
        self.synergy_posture_pub = rospy.Publisher('/cyberglove/calibrated/joint_states', JointState, queue_size=1)
        rospy.Subscriber("/srh/sendupdate", sendupdate, self._publish_remapped_postures_cb)

        self.mapping_matrix = _read_mapping()

        # to be used by trajectory controller sliders
        self.trajectory_state_sub = []
        # self.trajectory_state = []
        self.trajectory_state_slider_cb = []
        self.trajectory_pub = []
        self.trajectory_target = []

        self.pause_subscriber = False

        self._widget.synergyTypeCombo.currentIndexChanged.connect(
            self.on_synergy_combo_index_changed)
        self._widget.pcNumberCombo.currentIndexChanged.connect(
            self.on_pc_combo_index_changed)

        self._widget.synergyVectorViewer.setFontPointSize(8)
        self._widget.synergyVectorViewer.setFontFamily("DejaVu Sans Mono")

        self.posture_update.connect(self._update_synergy_viewer)

        self._widget.synergyTypeCombo.addItem("grasp")
        self._widget.synergyTypeCombo.addItem("sub1")
        self._widget.synergyTypeCombo.addItem("sub2")
        self._widget.synergyTypeCombo.addItem("sub3")
        self._widget.synergyTypeCombo.addItem("sub4")

        self.text_box = self._widget.synergyVectorViewer
        print self._widget.synergyVectorViewer.parent()

        self.hand_prefix = self._get_hand_prefix()
        print QThread.currentThreadId()
        print "finish initialization on : {}".format(QThread.currentThreadId())

    def _get_hand_prefix(self):
        hand_finder = HandFinder()
        if hand_finder._hand_e:
            hand_parameters = hand_finder.get_hand_parameters()
            key, hand_prefix = hand_parameters.joint_prefix.items()[0]
        elif hand_finder._hand_h:
            hand_prefix, value = hand_finder._hand_h_parameters.items()[0]
            hand_prefix = hand_prefix + "_"
        else:
            hand_prefix = ""
        return hand_prefix

    def _unregister(self):
        pass

    def shutdown_plugin(self):
        self._unregister()

    def save_settings(self, global_settings, perspective_settings):
        pass

    def restore_settings(self, global_settings, perspective_settings):
        pass

    def on_synergy_combo_index_changed(self):
        current_synergy_index = self._widget.synergyTypeCombo.currentIndex()
        self.current_synergy_type = self.synergy_types[current_synergy_index][0]
        self._widget.pcNumberCombo.clear()
        for i in range(1, self.synergy_types[current_synergy_index][1] + 1, 1):
            self._widget.pcNumberCombo.addItem(str(i))

    def on_pc_combo_index_changed(self):
        self.current_pc_num = self._widget.pcNumberCombo.currentIndex() + 1
        self.on_reload_button_cicked_()
        pass

    def on_robot_type_changed_(self):
        pass

    def on_reload_button_cicked_(self):
        """
        Clear existing slider widgets from layout
        Load the correct robot library
        Create and load the new slider widgets
        """
        self.pause_subscriber = True

        self._load_robot_description()
        controllers = self.get_current_controllers()

        self.joints = self._create_joints(controllers)

        self.synergy = self._create_synergy(controllers)

        self.delete_old_sliders_()

        # self._widget.sliderReleaseCheckBox.setCheckState(Qt.Unchecked)

        self.load_new_synergy_sliders_()

        # self.load_new_sliders_()

        self._update_synergy_viewer()

        self.pause_subscriber = False

    def on_refresh_button_cicked_(self):
        """
        Call refresh for every slider
        """
        for slider in self.sliders:
            slider.refresh()

    def on_slider_release_checkbox_clicked_(self, state):
        """
        Set tracking behaviour of each slider to false if checkbox is checked, true otherwise
        """

        if state == Qt.Checked:
            for slider in self.sliders:
                slider.set_new_slider_behaviour(False)
        else:
            for slider in self.sliders:
                slider.set_new_slider_behaviour(True)

    def delete_old_sliders_(self):
        """
        Clear existing slider widgets from layout
        Empty the slider list
        """
        for old_slider in self.sliders:
            self._widget.horizontalLayout.removeWidget(old_slider)
            old_slider.close()
            old_slider.deleteLater()

        self.sliders = []

        if (self.selection_slider is not None):
            self._widget.horizontalLayout.removeWidget(self.selection_slider)
            self.selection_slider.close()
            self.selection_slider.deleteLater()
            self.selection_slider = None

    def load_new_synergy_sliders_(self):
        self.sliders = list()
        for pc in self.synergy:
            slider = None
            slider_ui_file = os.path.join(rospkg.RosPack().get_path('gui_synergy_slider'), 'uis', 'Slider.ui')

            try:
                slider = EtherCATHandSynergySlider(pc, slider_ui_file, self,
                                                             self._widget.scrollAreaWidgetContents)
            except Exception, e:
                rospy.loginfo(e)

            if slider is not None:
                slider.setMaximumWidth(100)
                self._widget.horizontalLayout.addWidget(slider)
                self.sliders.append(slider)
            else:
                rospy.logerr("cannot add the slider object {}PC".format(pc.num))

    def load_new_sliders_(self):
        """
        Create the new slider widgets
        Load the new slider
        Put the slider in the list
        """
        self.sliders = list()
        for joint in self.joints:
            slider = None
            slider_ui_file = os.path.join(
                rospkg.RosPack().get_path('gui_synergy_slider'), 'uis', 'Slider.ui')
            try:
                if joint.controller.controller_category == "position_trajectory":
                    slider = EtherCATHandTrajectorySlider(
                        joint, slider_ui_file, self, self._widget.scrollAreaWidgetContents)
                else:
                    slider = EtherCATHandSlider(
                        joint, slider_ui_file, self, self._widget.scrollAreaWidgetContents)
            except Exception, e:
                rospy.loginfo(e)

            if slider is not None:
                slider.setMaximumWidth(100)
                # Load the new slider
                self._widget.horizontalLayout.addWidget(slider)
                # Put the slider in the list
                self.sliders.append(slider)

        # Create the slider to move all the selected joint sliders
        selection_slider_ui_file = os.path.join(
            rospkg.RosPack().get_path('gui_synergy_slider'), 'uis', 'SelectionSlider.ui')
        self.selection_slider = EtherCATSelectionSlider(
            "Change sel.", 0, 100, selection_slider_ui_file, self, self._widget.scrollAreaWidgetContents)

        self.selection_slider.setMaximumWidth(100)
        self._widget.horizontalLayout.addWidget(self.selection_slider)

    def get_current_controllers(self):
        """
        @return: list of current controllers with associated data
        """
        success = True
        list_controllers = rospy.ServiceProxy(
            'controller_manager/list_controllers', ListControllers)
        try:
            resp1 = list_controllers()
        except rospy.ServiceException:
            success = False

        if success:
            return [c for c in resp1.controller if c.state == "running"]
        else:
            rospy.loginfo(
                "Couldn't get list of controllers from controller_manager/list_controllers service")
            return []

    def _load_robot_description(self):
        """
        Load the description from the param named in the edit as an ET element.
        Sets self._robot_description_xml_root to the element.
        """
        name = "robot_description"  # self._widget.robot_description_edit.text()
        self._robot_description_xml_root = None
        try:
            xml = rospy.get_param(name)
            self._robot_description_xml_root = ET.fromstring(xml)
        except KeyError as e:
            rospy.logerr(
                "Failed to get robot description from param %s : %s" % (name, e))
            return
        except:
            raise

    def _get_joint_min_max_vel(self, jname):
        """Get the min and max from the robot description for a given joint."""
        root = self._robot_description_xml_root
        if root is not None:
            joint_type = root.findall(".joint[@name='" + jname + "']")[0].attrib['type']
            if joint_type == "continuous":
                limit = root.findall(".//joint[@name='" + jname + "']/limit")
                if limit is None or len(limit) == 0:
                    return (-math.pi,
                            math.pi,
                            3.0)  # A default speed
                else:
                    return (-math.pi,
                            math.pi,
                            float(limit[0].attrib['velocity']))
            else:
                limit = root.findall(".//joint[@name='" + jname + "']/limit")
                if limit is None or len(limit) == 0:
                    # Handles upper case joint names in the model. e.g. the E1
                    # shadowhand
                    limit = root.findall(
                        ".//joint[@name='" + jname.upper() + "']/limit")
                if limit is not None and len(limit) > 0:
                    return (float(limit[0].attrib['lower']),
                            float(limit[0].attrib['upper']),
                            float(limit[0].attrib['velocity']))
                else:
                    rospy.logerr("Limit not found for joint %s", jname)
        else:
            rospy.logerr("robot_description_xml_root == None")
        return (None, None, None)

    def _get_joint_min_max_vel_special(self, jname):
        if "J0" in jname:
            jname1 = jname.replace("J0", "J1")
            jname2 = jname.replace("J0", "J2")
            min1, max1, vel1 = self._get_joint_min_max_vel(jname1)
            min2, max2, vel2 = self._get_joint_min_max_vel(jname2)
            return (min1 + min2, max1 + max2, vel1 + vel2)
        else:
            return self._get_joint_min_max_vel(jname)

    def _create_synergy(self, controllers):
        principal_components = []
        pca_resource_path = os.path.join(rospkg.RosPack().get_path('gui_synergy_slider'), 'resource', 'taxonomy')

        synergy = None
        postures = None
        if self.current_synergy_type == "grasp":
            self.selected_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            synergy, postures = pca_utils.generate_toss_from_directory(pca_resource_path, pc_num=22, joint_num=22)
        elif "sub" in self.current_synergy_type:
            if self.current_synergy_type == "sub1":
                self.selected_joints = [0, 1, 2, 3]
            elif self.current_synergy_type == "sub2":
                self.selected_joints = [0, 1, 2, 3, 4, 5, 6, 10, 14, 18]
            elif self.current_synergy_type == "sub3":
                self.selected_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18]
            elif self.current_synergy_type == "sub4":
                self.selected_joints = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            synergy, postures = pca_utils.generate_subsynergy_from_directory(pca_resource_path,
                                                                             pc_num=len(self.selected_joints),
                                                                             joint_num=22,
                                                                             selected_joints=self.selected_joints)
        pc_score_range = pca_utils.principal_component_score_range(synergy, postures)

        self.synergy_score_change_sub = []
        self.current_pca_object = synergy
        self.current_synergy_scores = [0] * self.current_pc_num

        print pc_score_range
        for pc_num in range(1, self.current_pc_num + 1):
            principal_component = PrincipalComponent(num=pc_num, min=pc_score_range[pc_num-1][0],
                                                     max=pc_score_range[pc_num-1][1], vector=synergy.components_[pc_num-1])
            principal_components.append(principal_component)

            self.synergy_score_change_sub.append(rospy.Subscriber("synergy_score_change/pc{}".format(pc_num), Float64,
                                                                  self._publish_synergy_postures_cb,
                                                                  callback_args=pc_num))
        return principal_components

    def _update_synergy_viewer(self):
        print "update synergy on : {}".format(QThread.currentThreadId())
        self._widget.synergyVectorViewer.clear()
        self._widget.synergyVectorViewer.insertPlainText(
            pca_utils.print_synergy_vectors(vectors=self.current_pca_object.components_,
                                            selected_joints=self.selected_joints,
                                            pc_num=self.current_pc_num,
                                            mean_posture=self.current_pca_object.mean_))

        view = "\n---synergy description of SRH---\n"
        print self.selected_joints
        whole_joints_pca_components = np.zeros((len(self.selected_joints), 22))  # 22 is the number of joints of SRH
        # Care of the calculation of subsynergy with selected joints spaces
        for i, joint_idx in enumerate(self.selected_joints):
            whole_joints_pca_components[:, joint_idx] = self.current_pca_object.components_[:, i]

        mapped_synergy = np.dot(self.mapping_matrix.T, whole_joints_pca_components.T).T

        print mapped_synergy

        for joint in self.current_sr_joints_target:
            view += "{:>8},".format(joint.joint_name)
        view += '\n'
        for i in range(self.current_pc_num):
            for j in range(len(self.current_sr_joints_target)):
                view += "{:>8.2f},".format(mapped_synergy[i][j])
            view += "\n"

        view += "\n\n---current joint state---\n"
        for joint in self.current_sr_joints_target:
            view += "{:>8},".format(joint.joint_name)
        view += '\n'
        for joint in self.current_sr_joints_target:
            view += "{:>8.2f},".format(float(joint.joint_target))
        self._widget.synergyVectorViewer.insertPlainText(view)

    def _create_joints(self, controllers):
        joints = []
        trajectory_ctrl_joint_names = []
        self.trajectory_target = []
        self.trajectory_state_sub = []
        # self.trajectory_state = []
        self.trajectory_state_slider_cb = []
        self.trajectory_pub = []

        self.joint_position_controller_pub = dict()

        for controller in controllers:
            if controller.type == "position_controllers/JointTrajectoryController":
                for j_name in controller.claimed_resources[0].resources:
                    trajectory_ctrl_joint_names.append(j_name)

        for controller in controllers:
            if rospy.has_param(controller.name):
                ctrl_params = rospy.get_param(controller.name)
                controller_type = ctrl_params["type"]
                if controller_type in self.controller_state_types:
                    controller_state_type = self.controller_state_types[
                        controller_type][1]
                    controller_category = self.controller_state_types[
                        controller_type][0]

                    if controller_category == "position_trajectory":
                        # for a trajectory controller we will load a slider for every resource it manages
                        self.trajectory_target.append(JointTrajectory())
                        self.trajectory_state_sub.append(
                            rospy.Subscriber(controller.name + "/state", controller_state_type,
                                             self._trajectory_state_cb,
                                             callback_args=len(self.trajectory_state_sub)))

                        self.trajectory_state_slider_cb.append([])
                        self.trajectory_pub.append(
                            rospy.Publisher(controller.name + "/command", JointTrajectory, queue_size=1, latch=True))
                        for j_name in controller.claimed_resources[0].resources:
                            joint_controller = JointController(
                                controller.name, controller_type, controller_state_type, controller_category,
                                self.trajectory_state_slider_cb[
                                    len(self.trajectory_state_slider_cb) - 1],
                                self.trajectory_pub[
                                    len(self.trajectory_pub) - 1],
                                self.trajectory_target[len(self.trajectory_target) - 1])
                            rospy.loginfo(
                                "controller category: %s", controller_category)

                            if self._widget.joint_name_filter_edit.text() not in j_name:
                                continue

                            min, max, vel = self._get_joint_min_max_vel_special(
                                j_name)
                            joint = Joint(
                                j_name, min, max, vel, joint_controller)
                            joints.append(joint)
                    else:
                        joint_name = ctrl_params["joint"]
                        if joint_name in trajectory_ctrl_joint_names:
                            # These joints are controlled by the trajectory controller
                            continue

                        if "J0" in joint_name:  # xxJ0 are controlled by the by the trajectory controller xxJ1 and xxJ2
                            jname1 = joint_name.replace("J0", "J1")
                            jname2 = joint_name.replace("J0", "J2")
                            if jname1 in trajectory_ctrl_joint_names \
                                    and jname2 in trajectory_ctrl_joint_names:
                                continue

                        joint_controller = JointController(
                            controller.name, controller_type, controller_state_type, controller_category)
                        rospy.loginfo(
                            "controller category: %s", controller_category)

                        # if self._widget.joint_name_filter_edit.text() not in joint_name:
                        #     continue

                        min, max, vel = self._get_joint_min_max_vel_special(
                            joint_name)
                        joint = Joint(
                            joint_name, min, max, vel, joint_controller)
                        joints.append(joint)

                        if controller_category == "position":
                            self.joint_position_controller_pub[joint_name] = \
                                rospy.Publisher(joint.controller.name + "/command",
                                                Float64,
                                                queue_size=1,
                                                latch=True)

                else:
                    rospy.logwarn(
                        "Controller %s of type %s not supported", controller.name, controller_type)
                    continue
            else:
                rospy.logwarn(
                    "Parameters for controller %s not found", controller.name)
                continue

        return joints

    def _trajectory_state_cb(self, msg, index):
        if not self.pause_subscriber:
            if not self.trajectory_target[index].joint_names:  # Initialize the targets with the current position
                self.trajectory_target[index].joint_names = msg.joint_names
                point = JointTrajectoryPoint()
                point.positions = list(msg.actual.positions)  # This is a list for some reason? Should be tuple..
                point.velocities = [0] * len(msg.joint_names)
                point.time_from_start = rospy.Duration.from_sec(0.005)
                self.trajectory_target[index].points = [point]

            for cb in self.trajectory_state_slider_cb[index]:  # call the callbacks of the sliders in the list
                cb(msg)

    def _publish_synergy_postures_cb(self, msg, pc_num):
        self.current_synergy_scores[pc_num - 1] = msg.data
        next_posture = pca_utils.calculate_approximated_posture(mean_posture=self.current_pca_object.mean_,
                                                                pc_vectors=self.current_pca_object.components_,
                                                                scores=self.current_synergy_scores)
        if "sub" in self.current_synergy_type:
            all_joints = [0] * 22
            for i, joint_num in enumerate(self.selected_joints):
                all_joints[joint_num] = next_posture[i]
            next_posture = all_joints

        pub_msg = JointState()
        pub_msg.position = next_posture

        self.synergy_posture_pub.publish(pub_msg)

    def _publish_remapped_postures_cb(self, msg):
        print "publish posture on : {}".format(QThread.currentThreadId())
        pub_msg = Float64()
        self.current_sr_joints_target = msg.sendupdate_list
        for joint in msg.sendupdate_list:
            pub_msg.data = radians(float(joint.joint_target))
            self.joint_position_controller_pub["rh_{}".format(joint.joint_name)].publish(pub_msg)
        self.posture_update.emit()

