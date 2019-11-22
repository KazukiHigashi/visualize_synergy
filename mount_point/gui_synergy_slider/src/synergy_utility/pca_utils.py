# coding=utf-8
import numpy as np
import csv
from sklearn.decomposition import PCA

from synergy_utility import utils

# const
INT_MAX = 1 << 32

# constants in analysis
_GRASPS = 33
_JOINTS = 20
_PCA_N = 2
_TOSS_PCA_N = 2
_CLUSTER_N = 5

# constant parameter of graph
_LABEL = 18
_TICKS = 14
_FILE_TYPE = ".png"

_JOINT_NAME = ["T rot.", "T MCP", "T IP", "T abd.", "I MCP", "I PIP", "I DIP", "M MCP", "M PIP", "M DIP", "MI abd.",
               "R MCP", "R PIP", "R DIP", "RM abd.", "P MCP", "P PIP", "P DIP", "PR abd.", "Palm Arch"]
_DHAIBA_MODEL = ["CPDummy-Y", "TPP-X", "TDP-X", "TMCP-Z", "IPP-X", "IMP-X", "IDP-X", "MPP-X", "MMP-X", "MDP-X", "IMCP-Z"
    , "RPP-X", "RMP-X", "RDP-X", "RMCP-Z", "PPP-X", "PMP-X", "PDP-X", "PMCP-Z", "PMCP-Y"]


def generate_motion_along_with_pc_axis(taxonomy_mean, pc_axis, file_name, axis, coeff_range):
    coefficient_list = np.arange(start=coeff_range[0], step=(coeff_range[1] - coeff_range[0]) / 500,
                                 stop=coeff_range[1])

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(_DHAIBA_MODEL)
        for a in coefficient_list:
            writer.writerow((a * pc_axis[axis] + taxonomy_mean) * np.pi / 180)


def principal_component_score_range(pca_object, pca_resources):
    """

    :param pca_object: instance of PCA()
    :param pca_resources: np_array used to form the pca_object
    :return:
    """
    pc_scores = pca_object.transform(pca_resources)
    return [(min(pc_score), max(pc_score)) for pc_score in pc_scores.T]


def generate_toss_from_files(filenames=None):
    mixed_motions = []
    each_motions = []

    for file in filenames:
        motion = utils.parse_csv_task_motion(filename=file)
        mixed_motions.extend(motion)
        each_motions.append(motion)

    print("the number of posture to construct toss : " + str(len(mixed_motions)))

    toss_pca = PCA(n_components=_PCA_N)
    toss_pca.fit(mixed_motions)

    return toss_pca, mixed_motions


def generate_toss_from_directory(path="./", pc_num=20, joint_num=20):
    filelist = utils.enum_file_name(path)
    mixed_motions = []

    for file in filelist:
        motion = utils.parse_csv_task_motion(filename=file, joint_num=joint_num)
        mixed_motions.extend(motion)

    print("the number of posture to construct toss : " + str(len(mixed_motions)))

    toss_pca = PCA(n_components=pc_num)
    toss_pca.fit(mixed_motions)

    return toss_pca, mixed_motions


def comparison_pca_with_statistics(pca_list, task_filename_list):
    # read available joint angle range
    joint_range = utils.read_joint_range()

    approx_norm_err = [[] for i in range(len(pca_list))]
    for i in range(len(task_filename_list)):
        for j in range(len(pca_list)):
            motion_data = utils.parse_csv_task_motion(task_filename_list[i])
            motion_approx_error = motion_data - pca_list[j].inverse_transform(pca_list[j].transform(motion_data))
            approx_norm_err[j].append(np.abs(np.array(motion_approx_error) / joint_range).mean(axis=1).mean(axis=0))

    return np.array(approx_norm_err).mean(axis=1)  # np.median(np.array(approx_norm_err), axis=1)


def calculate_approximated_posture(mean_posture, pc_vectors, scores):
    posture = np.array(mean_posture)
    pc_vectors = np.array(pc_vectors)
    scores = np.array(scores)
    for i in range(len(scores)):
        posture += pc_vectors[i]*scores[i]
    return posture


