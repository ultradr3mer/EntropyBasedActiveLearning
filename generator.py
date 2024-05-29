import json
import math
import os.path
import random
from typing import io

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from classifier import KnnClassifier
from deepLearner import x_res, y_res, DeepLearner
import plotFunctions
from pointManager import PointManager
from util import normalize, if_then_else
from logger import instance as logger

res = x_res
blob_count = 24
dataset_filename = 'dataset.json'
uint16_max = 2 ** 16 - 1
acc_sum = 0
acc2_sum = 0


class Generator:
    def __init__(self, folder):
        self.rnd = random.Random('dfg43')
        self.classes = list(range(2))
        self.folder = folder

    def generate(self):
        center_list = []
        std_list = []
        for i in range(blob_count):
            next_center = [self.rnd_float(-18, 18), self.rnd_float(-18, 18)]
            next_std = self.rnd_float(1, 3)
            if any([1 for c, std in zip(center_list, std_list)
                    if np.linalg.norm(np.array(c) - np.array(next_center)) < (next_std + std)]):
                continue
            center_list.append(next_center)
            std_list.append(next_std)

        all_x, all_y = make_blobs(n_samples=self.rnd_int(300, 600),
                                  n_features=blob_count,
                                  centers=np.array(center_list),
                                  cluster_std=np.array(std_list))
        all_x = [list(p) for p in all_x]
        y_map = {i: self.rnd_int(0, 1) for i in range(blob_count)}
        y_map[0] = 0
        y_map[1] = 1
        all_y = [y_map[y] for y in all_y]

        return (all_x, all_y)  # train_test_split(all_x, all_y, test_size=self.rnd_float(0.1, 0.9), shuffle=True)

    def rnd_float(self, min, max):
        return self.rnd.random() * (max - min) + min

    def rnd_int(self, min, max):
        return self.rnd.randint(min, max)

    def read(self, json_string):
        data = json.loads(json_string)
        for item in data:
            x_labeled, x_unlabeled, y_labeled, y_unlabeled = item
            pass

    def load_or_create_dataset(self):
        def load():
            try:
                with open(dataset_filename, 'r') as f:
                    json_string = f.read()
                    return json.loads(json_string)
            except FileNotFoundError:
                data_set = []
                for i in range(2000):
                    x, y = self.generate()
                    data_set.append([list(x), y])
                json_string = json.dumps(data_set)
                with open(dataset_filename, 'w') as f:
                    f.write(json_string)
                return data_set

        return load()[:1000]


    def gen_training_histogram(self, item, labeled_i):
        pmgr = PointManager(item, labeled_i)

        x = np.dstack((pmgr.map_unlabeled, pmgr.map_label_0, pmgr.map_label_1))
        x = normalize(x)

        discovery_map = pmgr.histogram_class([pmgr.all_x[i] for i in pmgr.means_set(pmgr.all_x, 80)]).reshape(-1)

        map_unlabeled_flat = pmgr.map_unlabeled.reshape(-1)
        discovery_points = np.array([i for i, coord in enumerate(pmgr.point_coords) if map_unlabeled_flat[i] > 0])
        points_with_class = np.concatenate((pmgr.remaining_x, pmgr.remaining_y.reshape(-1, 1)), axis=1)
        probs = [[len([1 for p in b if p[2] == 0]), len([1 for p in b if p[2] == 1])]
                 for b in pmgr.point_bins(points_with_class)]
        probs = [if_then_else(sum(p) == 0, [1, 1], p) for p in probs]
        map_points = np.array([i for i in discovery_points if discovery_map[i] > 0.0])
        map_acc = np.array([np.average(pmgr.acc_for_point(pmgr.point_coords[i]), weights=[1, 0]) for i in map_points])
        # remaining_points = [i for i in discovery_points if discovery_map[i] == 0.0]

        discovery_hotmap = KnnClassifier(1, 3)
        discovery_hotmap.fit([pmgr.point_coords[i] for i in map_points],
                             [1.0 - c for c in map_acc])
        y_points = pmgr.gen_point_coords(y_res)
        y = discovery_hotmap.predict(y_points).reshape(y_res, y_res)
        y = normalize(y)

        p = pmgr.calc_prior()

        return x, y, p

    def save_as_image(self, x, filename):
        image = cv2.cvtColor(np.array(x * uint16_max, dtype='uint16'), cv2.CV_16U)
        cv2.imwrite(filename, image)
        pass

    def get_trainig_file_names(self, number):
        return {"x": f'{self.folder}/{number}_x.png',
                "y": f'{self.folder}/{number}_y.png',
                "p": f'{self.folder}/{number}_p.png'}

    def load_from_image(self, filename):
        img = cv2.imread(filename, -1)
        if img is None:
            raise FileNotFoundError(f"Image with {filename} could not be loaded.")
        return np.array(cv2.split(img), dtype=float) / uint16_max

    def gen_train_data(self, dataset, number, labeled_i=None):
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass

        file_names = self.get_trainig_file_names(number)
        if (os.path.isfile(file_names['x']) and
                os.path.isfile(file_names['y']) and
                os.path.isfile(file_names['p'])):
            return

        x, y, p = self.gen_training_histogram(dataset, labeled_i)
        self.save_as_image(x, file_names['x'])
        self.save_as_image(y, file_names['y'])
        self.save_as_image(p, file_names['p'])
        return

    def load_training_data(self, number):
        file_names = self.get_trainig_file_names(number)
        x = self.load_from_image(file_names['x'])[:3]
        y = self.load_from_image(file_names['y'])[0]
        p = self.load_from_image(file_names['p'])[0]
        return x, y, p


def load_labeled_i(stage_nr):
    with open(f'data{stage_nr}/labeled_indices.json', 'r') as f:
        return json.loads(f.read())


def pick_unlabeled(comm, stage_nr, whole_dataset, msg):
    learner = DeepLearner(f"data{stage_nr - 1}/model_weights.pth")

    pick_count = 0
    pick_success = 0

    # msg = msg[:50]

    step_indices_file_name = f"data{stage_nr}/labeled_indices.json"
    # combined_picks = None
    if not os.path.exists(step_indices_file_name):
        if stage_nr >= 2:
            last_indices = load_labeled_i(stage_nr - 1)
        else:
            last_indices = None

        worker_picks = []
        for nr, i in enumerate(msg):
            item = whole_dataset[i]

            if last_indices is not None:
                item_indices = last_indices[i]
            else:
                item_indices = None

            picked, success = learner.pick(item, item_indices)
            worker_picks.append(picked)

            pick_count += 1
            if success:
                pick_success += 1

            if nr % 5 == 0:
                success_rate = comm.gather(pick_success / pick_count, root=0)

                if success_rate is not None:
                    logger.info(f" progress  {int(nr / len(msg) * 100)}% success {int(np.average(success_rate) * 100)}%")

        success_rate = comm.gather(pick_success / pick_count, root=0)

        if success_rate is not None:
            logger.info(f" final success {int(np.average(success_rate) * 100)}%")

        all_picks = comm.gather(worker_picks, root=0)

        if all_picks is not None:
            combined_picks = [list([int(i) for i in p]) for worker_pick in all_picks for p in worker_pick]
            json_string = json.dumps(combined_picks)
            with open(step_indices_file_name, 'w') as f:
                f.write(json_string)
            return combined_picks
        else:
            return None

    elif comm.rank == 0:
        return load_labeled_i(stage_nr)

    return None


def get_stage_indices(stage_nr):
    with open(f'data{stage_nr}/labeled_indices.json', 'r') as f:
        return json.loads(f.read())


def run_stage(stage_nr):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    rank_count = comm.Get_size()

    stage_dir = f'data{stage_nr}'

    gen = Generator(os.path.join(stage_dir, 'training'))

    if my_rank == 0:
        logger.info(f'Starting stage {stage_nr}')
        logger.info(f'Create dataset')
        whole_dataset = gen.load_or_create_dataset()
        indices = np.array(range(len(whole_dataset)))
        msg_list = np.array_split(indices, rank_count)

        try:
            os.mkdir(stage_dir)
        except FileExistsError:
            pass
    else:
        whole_dataset = None
        msg_list = []

    msg = comm.scatter(msg_list, root=0)

    if stage_nr > 0:
        if whole_dataset is None:
            whole_dataset = gen.load_or_create_dataset()

        if my_rank == 0:
            logger.info(f'Starting picking {stage_nr}')

        indices = pick_unlabeled(comm, stage_nr, whole_dataset, msg)
    else:
        indices = None

    indices = comm.bcast(indices)
    whole_dataset = comm.bcast(whole_dataset)

    if my_rank == 0:
        logger.info(f'Starting training data generation {stage_nr}')

    total_processed = 0
    for nr, i in enumerate(msg):
        item = whole_dataset[i]
        if indices is not None:
            item_indices = indices[i]
        else:
            item_indices = None

        gen.gen_train_data(item, i, item_indices)

        if nr % 5 == 0:
            processed = comm.reduce(10, op=MPI.SUM)
            if processed is not None:
                total_processed += processed
                logger.info(f" progress  {int(total_processed / len(whole_dataset) * 100)}%")

    logger.info(f"Completed generation cycle {stage_nr}: rank {my_rank} / {rank_count}")

    if not os.path.exists(f'data{stage_nr}/model_weights.pth'):
        from deepTraining import DeepTrainer

        if my_rank == 0:
            logger.info(f'Starting model training {stage_nr}')
            trainer = DeepTrainer(f'data{stage_nr}')
            state = trainer.train()
            torch.save(state, f"{stage_dir}/model_weights.pth")

        if my_rank == 0:
            logger.info(f'Stage completed {stage_nr}')


if __name__ == '__main__':
    for i in range(10):
        run_stage(i)

