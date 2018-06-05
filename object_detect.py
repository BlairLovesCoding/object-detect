from preprocessing import DataLoader
from model import SGD, MLP
import numpy as np


def main():
    dataDir = '/Users/Blair/PycharmProjects/CSE547/data'
    dataTypes = ["train2014", "val2014", "test2014"]
    animal = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
    vehicle = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    category = animal + vehicle

    for cat in category[14:15]:

        for dataType in dataTypes[:2]:
            loader = DataLoader(dataDir, dataType, cat)
            loader.load(10000, 2)
            n_pos = loader.pos.shape[0]
            n_nega = loader.nega.shape[0]
            dt = np.row_stack([loader.pos, loader.nega])
            label_pos = np.ones([n_pos, 1])
            label_nega = np.zeros([n_nega, 1])
            label = np.row_stack([label_pos, label_nega])
            if dataType == dataTypes[0]:
                train_dt = dt
                train_lab = label
            # elif dataType == dataTypes[2]:
            #     test_dt = dt
            #     test_lab = label
            else:
                val_dt = dt
                val_lab = label

        loader = DataLoader(dataDir, dataTypes[2], cat)
        test_dt, test_lab = loader.load_test(2000)

        print("train data: ", train_dt.shape, "train label: ", train_lab.shape)
        print("val data: ", val_dt.shape, "val label: ", val_lab.shape)
        print("test data: ", test_dt.shape, "test label: ", test_lab.shape)
        num_pos = train_dt.shape[0] // 3
        iteration = 20 * train_dt.shape[0]
        plambda = 20
        batch_size = 20
        step_size = 1e-6
        # num_hidden = 100
        model = SGD(train_dt, train_lab, val_dt, val_lab, test_dt, test_lab, cat)
        # model_mlp = MLP(train_dt, train_lab, val_dt, val_lab, test_dt, test_lab, cat, num_hidden)

        for t in range(1):
            model.train(train_dt, iteration, plambda, batch_size, step_size)
            model.visualize(t)
            model.update.clear()
            model.l_train.clear()
            model.l_val.clear()
            model.ap_train.clear()
            model.ap_val.clear()
            # loader = DataLoader(dataDir, dataTypes[0], cat)
            # new_nega = loader.load_nega(10000, num_pos)
            # nega = np.row_stack([model.hard_nega, new_nega])
            # train_dt = np.row_stack([model.pos, nega])


if __name__ == '__main__':
    main()