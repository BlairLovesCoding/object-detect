import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import average_precision_score


class SGD:
    def __init__(self, train_dt, train_lab, val_dt, val_lab, test_dt, test_lab, category):
        self.train_dt = train_dt
        self.train_lab = train_lab
        self.val_dt = val_dt
        self.val_lab = val_lab
        self.test_dt = test_dt
        self.test_lab = test_lab
        self.update = []
        self.l_train = []
        self.l_val = []
        self.ap_train = []
        self.ap_val = []
        self.test_ap = 0
        self.cat = category

        n = self.train_dt.shape[0]
        m = n // 3
        d = self.train_dt.shape[1]
        k = self.train_lab.shape[1]
        self.pos = self.train_dt[:m]
        self.w = Variable(torch.randn(d, k) / 500, requires_grad=True)
        self.hard_nega = []

    def f(self, w, x):
        fx = x.mm(w)
        return fx

    def loss(self, plambda, w, x, y):
        y_hat = self.f(w,x)
        tmp = torch.mul(y, torch.log(1 + torch.exp(-y_hat))) + torch.mul(1 - y, torch.log(1 + torch.exp(y_hat)))
        n = y.shape[0]
        return torch.sum(tmp) / n + plambda * torch.norm(w) * torch.norm(w) / 2

    def train(self, train_sample, iteration, plambda, batch_size, step_size):
        start = time.time()
        val_dt = torch.FloatTensor(self.val_dt)
        val_lab = torch.FloatTensor(self.val_lab)
        test_dt = torch.FloatTensor(self.test_dt)
        test_lab = torch.FloatTensor(self.test_lab)

        n = train_sample.shape[0]
        m = n // 3
        d = train_sample.shape[1]
        k = self.train_lab.shape[1]
        # w = Variable(torch.randn(d, k) / 500, requires_grad=True)
        w = self.w

        iter_per_epoch = n // batch_size
        epoch = (iteration - 1) // iter_per_epoch + 1
        count_iter = 0

        for j in range(epoch):
            combine_train = np.column_stack((train_sample, self.train_lab))
            np.random.shuffle(combine_train)
            train_dt = torch.FloatTensor(combine_train[:, :d])
            train_lab = torch.FloatTensor(combine_train[:, d:])
            for i in range(iter_per_epoch):
                if count_iter >= iteration:
                    break
                count_iter = count_iter + 1
                if count_iter % ((n / batch_size) // 2) == 0:
                    self.update.append(count_iter / (n / batch_size / 2))
                    self.l_train.append(self.loss(plambda, w.data, train_dt, train_lab))
                    self.l_val.append(self.loss(plambda, w.data, val_dt, val_lab))
                    #
                    # print("linear - iter: %d, train loss: %.4f, val loss: %.4f" %
                    #      (count_iter, self.l_train[-1], self.l_val[-1]))
                    train_sc = self.f(w.data, train_dt).tolist()
                    train_pdt = [s[0] for (s,l) in zip(train_sc, train_lab.tolist()) if s[0] > 0.5 or l == 1]
                    train_act = [l for (s,l) in zip(train_sc, train_lab.tolist()) if s[0] > 0.5 or l == 1]
                    val_sc = self.f(w.data, val_dt).tolist()
                    val_pdt = [s[0] for (s,l) in zip(val_sc, val_lab.tolist()) if s[0] > 0.5 or l == 1]
                    val_act = [l for (s,l) in zip(val_sc, val_lab.tolist()) if s[0] > 0.5 or l == 1]
                    self.ap_train.append(average_precision_score(train_act, train_pdt, average='micro'))
                    self.ap_val.append(average_precision_score(val_act, val_pdt, average='micro'))

                    print("linear - train ap: %.4f, val ap: %.4f" %
                         (self.ap_train[-1], self.ap_val[-1]))

                train_dt_batch = Variable(train_dt[(i * batch_size): ((i + 1) * batch_size)], requires_grad=False)
                train_lab_batch = Variable(train_lab[(i * batch_size): ((i + 1) * batch_size)], requires_grad=False)

                _loss = self.loss(plambda, w, train_dt_batch, train_lab_batch)
                _loss.backward()

                w.data = w.data - w.grad.data * step_size
                w.grad.data.zero_()

        end = time.time()
        print("%d seconds, linear training done" % (end - start))
        test_sc = self.f(w.data, test_dt).tolist()
        test_pdt = [s[0] for (s,l) in zip(test_sc, test_lab.tolist()) if s[0] > 0.5 or l == 1]
        test_act = [l for (s,l) in zip(test_sc, test_lab.tolist()) if s[0] > 0.5 or l == 1]
        self.test_ap = average_precision_score(test_act, test_pdt, average='micro')
        print(self.cat, " - linear - average test ap: %.4f" % self.test_ap)

        self.w = w
        nega = self.train_dt[m:]
        nega_dt = torch.FloatTensor(nega)
        predict = torch.abs(self.f(w.data, nega_dt)).view(n - m)
        # print(predict.shape)
        res, ind = predict.topk(m)
        # print(res)
        self.hard_nega = nega[ind]

    def visualize(self, iter):
        plt.figure(2*iter)
        plt.title("Logistic Regression - {} L(w) vs Iteration".format(self.cat))
        plt.xlabel("Epoch number")
        plt.ylabel("L(w)")
        train_plot, = plt.plot(self.update, self.l_train, label='train set')
        val_plot, = plt.plot(self.update, self.l_val, label='validate set')
        plt.legend([train_plot, val_plot], ['train set', 'validate set'])
        plt.savefig("{}_{}_loss.eps".format(self.cat, iter), format='eps')

        plt.figure(2*iter + 1)
        plt.title("Logistic Regression - {} AP Score vs Iteration".format(self.cat))
        plt.xlabel("Epoch number")
        plt.ylabel("AP Score")
        train_plot, = plt.plot(self.update, self.ap_train, label='train set')
        val_plot, = plt.plot(self.update, self.ap_val, label='validate set')
        plt.legend([train_plot, val_plot], ['train set', 'validate set'])
        plt.savefig("{}_{}_ap.eps".format(self.cat, iter), format='eps')


class MLP:
    def __init__(self, train_dt, train_lab, val_dt, val_lab, test_dt, test_lab, category, num_hidden):
        self.train_dt = train_dt
        self.train_lab = train_lab
        self.val_dt = val_dt
        self.val_lab = val_lab
        self.test_dt = test_dt
        self.test_lab = test_lab
        self.update = []
        self.l_train = []
        self.l_val = []
        self.ap_train = []
        self.ap_val = []
        self.test_ap = 0
        self.cat = category
        self.num_hidden = num_hidden

        n = self.train_dt.shape[0]
        m = n // 3
        d = self.train_dt.shape[1]
        k = self.train_lab.shape[1]
        self.pos = self.train_dt[:m]
        self.w1 = Variable(torch.randn(num_hidden, d) / 500, requires_grad=True)
        self.w2 = Variable(torch.randn(num_hidden, k) / 500, requires_grad=True)
        self.hard_nega = []

    def f_mlp(self, w1, w2, x):
        fx = x.mm(w1.t()).clamp(min=0).mm(w2)
        return fx

    def loss_mlp(self, plambda, w1, w2, x, y):
        y_hat = self.f_mlp(w1, w2, x)
        tmp = torch.mul(y, torch.log(1 + torch.exp(-y_hat))) + torch.mul(1 - y, torch.log(1 + torch.exp(y_hat)))
        n = y.shape[0]
        return torch.sum(tmp) / n + plambda * (torch.norm(w1) * torch.norm(w1) + torch.norm(w2) * torch.norm(w2)) / 2

    def train(self, train_sample, iteration, plambda, batch_size, step_size):
        start = time.time()
        val_dt = torch.FloatTensor(self.val_dt)
        val_lab = torch.FloatTensor(self.val_lab)
        test_dt = torch.FloatTensor(self.test_dt)
        test_lab = torch.FloatTensor(self.test_lab)

        n = train_sample.shape[0]
        m = n // 3
        d = train_sample.shape[1]
        k = self.train_lab.shape[1]
        w1 = self.w1
        w2 = self.w2

        iter_per_epoch = n // batch_size
        epoch = (iteration - 1) // iter_per_epoch + 1
        count_iter = 0

        for j in range(epoch):
            combine_train = np.column_stack((train_sample, self.train_lab))
            np.random.shuffle(combine_train)
            train_dt = torch.FloatTensor(combine_train[:, :d])
            train_lab = torch.FloatTensor(combine_train[:, d:])
            for i in range(iter_per_epoch):
                if count_iter >= iteration:
                    break
                count_iter = count_iter + 1
                if count_iter % ((n / batch_size) // 2) == 0:
                    self.update.append(count_iter / (n / batch_size / 2))
                    self.l_train.append(self.loss_mlp(plambda, w1.data, w2.data, train_dt, train_lab))
                    self.l_val.append(self.loss_mlp(plambda, w1.data, w2.data, val_dt, val_lab))

                    # print("mlp - iter: %d, train loss: %.4f, val loss: %.4f" %
                    #      (count_iter, self.l_train[-1], self.l_val[-1]))

                    self.ap_train.append(average_precision_score(train_lab, self.f_mlp(w1.data, w2.data, train_dt), average='micro'))
                    self.ap_val.append(average_precision_score(val_lab, self.f_mlp(w1.data, w2.data, val_dt), average='micro'))

                    # print("mlp - train ap: %.4f, val ap: %.4f" %
                    #      (self.ap_train[-1], self.ap_val[-1]))

                train_dt_batch = Variable(train_dt[(i * batch_size): ((i + 1) * batch_size)], requires_grad=False)
                train_lab_batch = Variable(train_lab[(i * batch_size): ((i + 1) * batch_size)], requires_grad=False)

                _loss = self.loss_mlp(plambda, w1, w2, train_dt_batch, train_lab_batch)
                _loss.backward()

                w1.data = w1.data - w1.grad.data * step_size
                w2.data = w2.data - w2.grad.data * step_size
                w1.grad.data.zero_()
                w2.grad.data.zero_()

        end = time.time()
        print("%d seconds, mlp training done" % (end - start))
        self.test_ap = average_precision_score(test_lab, self.f_mlp(w1.data, w2.data, test_dt), average='micro')
        print(self.cat, " - mlp - average test ap: %.4f" % self.test_ap)

        self.w1 = w1
        self.w2 = w2
        nega = self.train_dt[m:]
        nega_dt = torch.FloatTensor(nega)
        predict = torch.abs(self.f_mlp(w1.data, w2.data, nega_dt)).view(n - m)
        # print(predict.shape)
        res, ind = predict.topk(m)
        # print(res)
        self.hard_nega = nega[ind]

    def visualize(self, iter):
        plt.figure(2*iter)
        plt.title("HW3 P4 {} L(w) vs Iteration (MLP)".format(self.cat))
        plt.xlabel("Epoch number")
        plt.ylabel("L(w)")
        train_plot, = plt.plot(self.update, self.l_train, label='train set')
        val_plot, = plt.plot(self.update, self.l_val, label='validate set')
        plt.legend([train_plot, val_plot], ['train set', 'validate set'])
        plt.savefig("{}_mlp_{}_loss.eps".format(self.cat, iter), format='eps')

        plt.figure(2*iter + 1)
        plt.title("HW3 P4 {} AP Score vs Iteration (MLP)".format(self.cat))
        plt.xlabel("Epoch number")
        plt.ylabel("AP Score")
        train_plot, = plt.plot(self.update, self.ap_train, label='train set')
        val_plot, = plt.plot(self.update, self.ap_val, label='validate set')
        plt.legend([train_plot, val_plot], ['train set', 'validate set'])
        plt.savefig("{}_mlp_{}_ap.eps".format(self.cat, iter), format='eps')