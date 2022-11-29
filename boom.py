from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import Dataset
from qm9_main import Embed
import argparse
import utils
import json

parser = argparse.ArgumentParser(description='QM9 Adjusted Size')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def load_data(data_dir="./data"):

    trainset = QM9Dataset(fold='train')

    testset = QM9Dataset(fold='test')

    return trainset, testset

class Net(nn.Module):
    def __init__(
            self,
            l1 = 100,
            l2 = 50,
            l3 = 5,
            l4 = 5,
            l5 = 5,
            l6 = 5,
            l7 = 5,
            l8 = 5
    ):
        super(Net, self).__init__()
        self.embed = Embed()
        self.l1 = torch.tensor(l1).to(torch.int32)
        self.l2 = torch.tensor(l2).to(torch.int32)
        self.l3 = torch.tensor(l3).to(torch.int32)
        self.l4 = torch.tensor(l4).to(torch.int32)
        self.l5 = torch.tensor(l5).to(torch.int32)
        self.l6 = torch.tensor(l6).to(torch.int32)
        self.l7 = torch.tensor(l7).to(torch.int32)
        self.l8 = torch.tensor(l8).to(torch.int32)

        self.bn1 = nn.BatchNorm1d(num_features=l1)
        self.bn2 = nn.BatchNorm1d(num_features=l2)
        self.bn3 = nn.BatchNorm1d(num_features=l3)
        self.bn4 = nn.BatchNorm1d(num_features=l4)
        self.bn5 = nn.BatchNorm1d(num_features=l5)
        self.bn6 = nn.BatchNorm1d(num_features=l6)
        self.bn7 = nn.BatchNorm1d(num_features=l7)
        self.bn8 = nn.BatchNorm1d(num_features=l8)

        self.nonlin = F.relu
        self.first = 2*3*29 + 1
        self.dense0 = nn.Linear(self.first, self.l1)
        self.dense1 = nn.Linear(self.l1, self.l2)
        self.dense2 = nn.Linear(self.l2, self.l3)
        self.dense3 = nn.Linear(self.l3, self.l4)
        self.dense4 = nn.Linear(self.l4, self.l5)
        self.dense5 = nn.Linear(self.l5, self.l6)
        self.dense6 = nn.Linear(self.l6, self.l7)
        self.dense7 = nn.Linear(self.l7, self.l8)
        self.output = nn.Linear(self.l8, 1)


    def forward(self, X, **kwargs):
        X = self.embed(X)
        X = self.nonlin(self.bn1(self.dense0(X)))
        X = self.nonlin(self.bn2(self.dense1(X)))
        X = self.nonlin(self.bn3(self.dense2(X)))
        X = self.nonlin(self.bn4(self.dense3(X)))
        X = self.nonlin(self.bn5(self.dense4(X)))
        X = self.nonlin(self.bn6(self.dense5(X)))
        X = self.nonlin(self.bn7(self.dense6(X)))
        X = self.nonlin(self.bn8(self.dense7(X)))
        X = self.output(X)
        return X

def train_qm9(config, checkpoint_dir=None, data_dir=None, device="cpu"):
    net = Net(config["l1"], config["l2"], config["l3"],config["l4"])

    net.train()
    #assert(torch.cuda.is_available())
    net.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], betas=(0.9, 0.999), eps=1e-08)
    #optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8) #0.8 standard
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        drop_last=True)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        drop_last=True)

    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.squeeze().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config["clip"])
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.squeeze().to(device)

                outputs = net(inputs).squeeze()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        torch.save((net.state_dict(), optimizer.state_dict()), '/home/snirhordan/qm9results/zpve_dist_{}_{}_{}.pt'.format(config["l1"], config["l2"], config["l3"], config["l4"],config["l5"], config["clip"]))


        ray.tune.report(loss=(val_loss /  val_steps))
    print("Finished Training")

def test_loss(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)
    test_loss = 0 
    test_steps = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for data in testloader:
            instances, labels = data
            instances, labels = instances.to(device), labels.to(device)
            outputs = net(instances)
            loss = criterion(outputs, labels)
            test_loss += loss.cpu().numpy()
            test_steps += 1

    return test_loss / test_steps

def main(num_samples=10, max_num_epochs=50, gpus_per_trial=1):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": ray.tune.sample_from(lambda _: 2 ** np.random.randint(9, 14)),
        "l2": ray.tune.sample_from(lambda _: 2 ** np.random.randint(9, 14)),
        "l3": ray.tune.sample_from(lambda _: 2 ** np.random.randint(9, 14)),
        "l4": ray.tune.sample_from(lambda _: 2 ** np.random.randint(8, 12)),
        "l5": ray.tune.sample_from(lambda _: 2 ** np.random.randint(8, 12)),
        "l6": ray.tune.sample_from(lambda _: 2 ** np.random.randint(3,9)),
        "l7": ray.tune.sample_from(lambda _: 2 ** np.random.randint(3,9)),
        "l8": ray.tune.sample_from(lambda _: 2 ** np.random.randint(2,5)),
        "lr": ray.tune.loguniform(1e-6, 1e-2),
        "batch_size": ray.tune.choice([  32, 128]),
        "clip" : ray.tune.loguniform(50, 150)
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=20,
        reduction_factor=2)
    reporter = CLIReporter(
        #parameter_columns=["l1", "l2", "l3", "lr", "batch_size"],
        metric_columns=["loss",  "training_iteration"])
    ray.init(num_cpus =12, num_gpus=gpus_per_trial)
    result = ray.tune.run(
        partial(train_qm9, data_dir=data_dir, device='cuda'),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    #print("Best trial final validation accuracy: {}".format(
    #    best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"], best_trial.config["l3"], best_trial.config["l4"],best_trial.config["l5"],best_trial.config["l6"],best_trial.config["l7"],best_trial.config["l8"] )
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value # what is this?
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_loss = test_loss(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_loss))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=40, max_num_epochs=500, gpus_per_trial=2) #gpus_per_trial=1?