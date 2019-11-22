# coding=utf-8
import numpy as np
import csv
import os
import sys
from sklearn.decomposition import PCA


def make_dir_to_path(path):
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def read_joint_range():
    """
    各関節の可動範囲を計算
    :return: 各関節の可動範囲
    """
    joint_range = []
    with open("output/joint_range.csv") as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                joint_range.append(float(row[1]) - float(row[0]))
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format("a", reader.line_num, e))
    return joint_range


def parse_csv_mean_posture(path="data/", posture_num=33, joint_num=20):
    """
    Grasp Taxonomy用
    :param path: 動作データの格納ディレクトリへのパス
    :param posture_num: 動作データの個数
    :param joint_num: 使用する関節の個数（先頭から数える）
    :return: @path以下の動作データ(.csv)ファイルごとに、時間平均姿勢をリストで返す
    """

    pos_data = np.array([])
    for i in range(1, posture_num + 1):
        posture = np.zeros(joint_num)
        filename = path + str(i) + '.csv'
        with open(filename) as f:
            reader = csv.reader(f)
            try:
                for row in reader:
                    posture += np.array([float(i) * 180.0 / np.pi for i in row[1:joint_num + 1]])
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
        posture /= sum(1 for line in open(filename))

        if i == 1:
            pos_data = [posture]
        else:
            pos_data = np.append(pos_data, [posture], axis=0)
    return pos_data


def parse_any_csv(filename_list, joint_num=20):
    """
    :param filename_list: 結合する動作データファイルへの絶対パスリスト
    :return: 絶対パスで指定された全ての動作の重複しない姿勢データ(.csv)ファイルを全て結合する。
    """
    posture_list = []

    for name in filename_list:
        for pos in parse_csv_task_motion(name, joint_num=joint_num):
            posture_list.append(pos)
    # print("The number of postures: " + str(len(postures)))

    return np.array(posture_list)


def parse_csv_task_motion(filename="", joint_num=20):
    """
    :param filename: パースする動作データファイル(.csv）への絶対パス
    :param joint_num: 使用する関節の個数（先頭から数える）
    :return: 動作データファイル中の重複しない姿勢データをリストにパース
    """

    # parse all data
    pos = []
    with open(filename) as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                pos.append([float(i) for i in row[1:joint_num + 1]])
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
    # eliminate same data
    picked = []
    for row in pos:
        picked.append(row)
        if len(picked) != 1:
            if picked[-1] == picked[-2]:
                picked.pop()

    return np.array(picked)


def parse_csv(filename="", joint_num=20, include_timestamp=False):
    # parse all data
    pos = []
    with open(filename) as f:
        reader = csv.reader(f)
        try:
            for row in reader:
                if include_timestamp:
                    tmp = [float(row[0])]
                    tmp.extend([float(i) for i in row[1:joint_num + 1]])
                    pos.append(tmp)
                else:
                    pos.append([float(i) for i in row[1:joint_num + 1]])
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

    return pos


def enum_file_name(path="./"):
    """
    ファイル名の列挙
    :param path: 列挙したいファイルのあるディレクトリへのパス
    :return: @path以下のファイルへの絶対パスリスト
    """

    files = []
    current_list = os.listdir(path)

    for elem in current_list:
        full_path = path + "/" + elem

        if os.path.isdir(full_path):
            files.extend(enum_file_name(path=full_path))
        elif ".DS_Store" in full_path:
            continue
        else:
            files.append(full_path)
    # print("Number of Files : " + str(len(files)))

    return files


def enum_task_directory(path="./"):
    """
    @path以下の子にディレクトリを含まないディレクトリの列挙
    :param path: 列挙したい上記のディレクトリを含むディレクトリのパス
    :return: @path以下の上記のディレクトリへの絶対パスリスト
    """

    current_list = os.listdir(path)
    directories = []
    for elem in current_list:
        full_path = path + "/" + elem
        if os.path.isdir(full_path):
            directories.extend(enum_task_directory(path=full_path))
        else:
            directories.append(path)

    return list(set(directories))


def print_something(print_data, fitted_pca, pca_num, joint_num):
    print("33個の把持姿勢から得た各関節変位 : http://grasp.xief.net/")
    print(print_data)
    print()

    print("i(1~" + str(pca_num) + ")番目の主成分のみによる寄与率")
    print(fitted_pca.explained_variance_ratio_)
    print()
    print(str(pca_num) + "番目までの主成分の累積寄与率")
    print(sum(fitted_pca.explained_variance_ratio_))
    print()
    print("i番目の主成分の軸")
    print(fitted_pca.components_)
    print()
    print("元データ(" + str(joint_num) + "次元)を、主成分空間(" + str(pca_num) + "次元)に射影")
    print(fitted_pca.transform(print_data))


def calc_error_by_toss(eval_motion_file_name, task_pca, task_name):
    """
    タスクごとに構成したシナジーで、あるタスク中の姿勢の近似誤差を計算
    :param eval_motion_file_name: 評価するタスク
    :param task_pca: タスクごとに構成したシナジー(PCAオブジェクト)のリスト
    :param task_name: task_pcaに対応したタスクのディレクトリの絶対パス
    :return: 評価タスク中の姿勢ごとに計算した、近似誤差のリスト
    """

    motion = parse_csv_task_motion(eval_motion_file_name)
    joint_range = read_joint_range()

    # print("task_pca : " + str(len(task_pca)))
    # print("task_name : " + str(len(task_name)))

    approx_norm_err = 0.0

    for i in range(len(task_name)):
        # print(motion_file_name + " in " + task_name[i])
        if task_name[i] in eval_motion_file_name:
            motion_approx_error = motion - task_pca[i].inverse_transform(task_pca[i].transform(motion))
            approx_norm_err = np.abs(np.array(motion_approx_error) / joint_range).mean(axis=1).mean(axis=0)

    return approx_norm_err


def calc_norm_approx_error(pca, eval_motion):
    """

    :param pca: PCAオブジェクト
    :param eval_motion: 評価する動作データファイル
    :return: 評価タスクの各姿勢毎に計算した、近似誤差のリスト
    """
    joint_range = read_joint_range()

    motion_approx_error = eval_motion - pca.inverse_transform(pca.transform(eval_motion))
    return np.abs(np.array(motion_approx_error) / joint_range).mean(axis=1)


def generate_all_toss(directory_name_list):
    """

    :param directory_name_list: ディレクトリの絶対パスのリスト
    :return: ディレクトリごとに構成したシナジー(PCAオブジェクト)
    """
    toss_list = []

    for directory in directory_name_list:
        toss_list.append(generate_toss_from_directory(directory)[0])

    print(toss_list)
    return toss_list


def generate_toss_from_files(filename_list=None, pca_num=5):
    """
    :param filename_list: 指定されたファイルへのパスのリスト
    :param pca_num: シナジーの主成分数
    :return: 指定されたファイルから構成したPCAオブジェクト
    """

    mixed_motions = []
    each_motions = []

    for file in filename_list:
        motion = parse_csv_task_motion(filename=file)
        mixed_motions.extend(motion)
        each_motions.append(motion)

    print("the number of posture to construct toss : " + str(len(mixed_motions)))

    toss_pca = PCA(n_components=pca_num)
    toss_pca.fit(mixed_motions)

    return toss_pca, mixed_motions


def generate_toss_from_directory(path="./", pca_num=5):
    """
    :param path: シナジーを構成するディレクトリへのパス
    :param pca_num: シナジーの主成分数
    :return: 指定されたディレクトリ内のファイルから構成したPCAオブジェクト
    """
    filelist = enum_file_name(path)
    mixed_motions = []

    for file in filelist:
        motion = parse_csv_task_motion(filename=file)
        mixed_motions.extend(motion)

    print("the number of posture to construct toss : " + str(len(mixed_motions)))

    toss_pca = PCA(n_components=pca_num)
    toss_pca.fit(mixed_motions)

    return toss_pca, mixed_motions


def print_status_of_wrist(fitted_pca, motions, pc_num, joint_num=20):
    """

    :param fitted_pca: 学習済みのPCAオブジェクト
    :param motions:
    :param pc_num:
    :return:
    """
    trans = fitted_pca.transform(motions).T
    print(fitted_pca.components_)
    coeff_range = [min(trans[pc_num - 1]), max(trans[pc_num - 1])]
    wrist_yaw = fitted_pca.components_[pc_num - 1][joint_num - 1]
    coefficient_list = np.arange(start=coeff_range[0], step=(coeff_range[1] - coeff_range[0]) / 10, stop=coeff_range[1])

    # (a*pc_axis[axis]+taxonomy_mean)*np.pi/180
    print(coefficient_list * wrist_yaw + np.mean(motions, axis=0)[joint_num - 1])


def calc_joint_range(path="./", joint_num=20):
    """

    :param path: 関節角度を記録するための最も上位のディレクトリ
    :param joint_num: 関節数
    :return: 各関節における角度の最大値と最小値
    """
    files = enum_file_name(path)
    joint_range = [[1 << 32, -(1 << 32)] for i in range(joint_num)]

    postures = parse_any_csv(files)

    for i in range(joint_num):
        # print(postures[:, i])
        joint_range[i][0] = min(postures[:, i])
        joint_range[i][1] = max(postures[:, i])

    with open("output/joint_range.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        for a in joint_range:
            writer.writerow(a)

    return joint_range