import timeit
from datetime import datetime
import socket
import glob
from tqdm import tqdm
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset_light_dark_new_single_DARK import vehicle_state_dataset
from Networks.Transformer_dark import dark_net, pred_head, dark_net_LSTM, dark_net_informer, dark_net_tcn
# from Networks.Transformer_dark_baselinenet import dark_net, pred_head

from Networks.MultiTaskLossWrapper import cylinder_loss, MultiTaskLossWrapper_3
from utils_f import classification_results, metric
from Film_configs import Configs_pred, Configs_recons, Configs_pred_autoformer, Configs_recons_autoformer

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Device being used:", device)

nEpochs = 300           # Number of epochs for training
resume_epoch = 0        # Default is 0, change if want to resume
useTest = True          # See evolution of the test set when training
nTestInterval = 1       # Run on test set every nTestInterval epochs
snapshot = 350           # Store a model every snapshot epochs
lr = 1e-4               # Learning rate

num_classes = 2         # Number of total vehicle states: 0 normal, 1 fault
num_fault = 9           # Number of fault types in area: 0 normal, 1 hardover, 2 erratic, 3 spike, 4 drift

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print('save_dir_root is:', save_dir_root)

exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + '0')

modelName = 'Darknet'
modelType = modelName + '_TF'
saveName = modelName

rand_flg = True

print('savename {}'.format(saveName))


def train_model(modelType=modelType, save_dir=save_dir, num_classes=num_classes, num_fault=num_fault,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    print('Training in progress')
    print('modelType:', modelType)

    dark_model = dark_net_tcn(device=device).to(device)
    # dark_model = dark_net_informer(Configs_pred_autoformer, Configs_recons_autoformer, device=device).to(device)

    pred_model = pred_head(num_fault=num_fault).to(device)

    mt_loss = MultiTaskLossWrapper_3(task_num=3)

    train_params = [
        {'params': dark_model.parameters(), 'lr': lr},
        {'params': pred_model.parameters(), 'lr': lr},
        {'params': mt_loss.parameters(), 'lr': lr},
    ]

    print('train_params', train_params)

    print('Total params: %.2fM' % (sum(p.numel() for p in dark_model.parameters()) * 4 / 1000000.0))

    criterion = nn.MSELoss()

    optimizer = optim.Adam(train_params, lr=lr, betas=(0.5, 0.999), weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU

        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        optimizer.load_state_dict(checkpoint['opt_dict'])


    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    print('tensorboard log_dir', log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    bat_size = 128

    train_dataset = vehicle_state_dataset(split='train')
    test_dataset = vehicle_state_dataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=bat_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bat_size, shuffle=True, num_workers=4, pin_memory=True)

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)

    min_test_reg = 3.0
    max_test_cla = 0.5
    min_test_recons = 0.5

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        print('epoch {}'.format(epoch))
        start_time = timeit.default_timer()
        print('start time', start_time)

        optimizer.step()
        scheduler.step()

        dark_model.train()
        pred_model.train()

        # reset the running loss and corrects
        running_loss = 0.0
        running_cylinder_corrects = 0.0
        running_loss_cylinder_reg = 0.0
        running_loss_cylinder_recons = 0.0

        for total_info in tqdm(train_dataloader):

            dark_data = total_info[0]
            dark_targ = total_info[1]
            dark_recons_tg = total_info[2]
            dark_label = total_info[3]

            dim14_data = dark_data.to(device)
            dim14_targ = dark_targ.to(device)
            dark_recons_tg = dark_recons_tg.to(device)

            cylinder_array_lab = dark_label.long()
            cylinder_array_lab = Variable(cylinder_array_lab, requires_grad=False).to(device)

            # decoder input
            if Configs_pred_autoformer.padding == 0:
                dec_inp = torch.zeros([dim14_targ.shape[0], Configs_pred_autoformer.pred_len, dim14_targ.shape[-1]]).float().to(device)
            elif Configs_pred_autoformer.padding == 1:
                dec_inp = torch.ones([dim14_targ.shape[0], Configs_pred_autoformer.pred_len, dim14_targ.shape[-1]]).float().to(device)

            dec_inp = torch.cat([dim14_data[:, :Configs_pred_autoformer.label_len, :], dec_inp], dim=1).float().to(device)

            # pred_feat, recons_feat = dark_model(dim14_data, dim14_data, dec_inp, dec_inp)
            pred_feat, recons_feat = dark_model(dim14_data, dim14_targ)

            output, recons_output, cylinder_sig_type = pred_model(pred_feat, recons_feat)

            loss_reg = criterion(output,dim14_targ)
            loss_recons = criterion(recons_output,dark_recons_tg)
            loss_cylinder = cylinder_loss(cylinder_sig_type, cylinder_array_lab)

            cylinder_corrects, predsc, labsc = classification_results(cylinder_sig_type, cylinder_array_lab)

            loss_tot, param = mt_loss(loss_reg, loss_recons, loss_cylinder)

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

            running_loss += loss_tot.item() * dim14_data.size(0)
            running_cylinder_corrects += torch.mean(cylinder_corrects)
            running_loss_cylinder_reg += loss_reg.item() * dark_data.size(0)
            running_loss_cylinder_recons += loss_recons.item() * dark_data.size(0)

        epoch_loss = running_loss / train_size
        epoch_loss_cylinder_reg = running_loss_cylinder_reg / train_size
        epoch_loss_cylinder_recons = running_loss_cylinder_recons / train_size
        epoch_acc_cylinder = running_cylinder_corrects / train_size

        print('Training epoch_loss', epoch_loss_cylinder_reg, 'running_cylinder_corrects', epoch_acc_cylinder, 'loss_cylinder_recons', epoch_loss_cylinder_recons)

        writer.add_scalar('data/training_epoch_loss', epoch_loss, epoch)
        writer.add_scalar('data/training_regression_loss', epoch_loss_cylinder_reg, epoch)
        writer.add_scalar('data/training_classification_accuracy', epoch_acc_cylinder, epoch)
        writer.add_scalar('data/training_reconstruction_loss', epoch_loss_cylinder_recons, epoch)

        stop_time = timeit.default_timer()
        print("Execution time train: " + str(stop_time - start_time) + "\n")

        if useTest and epoch % test_interval == (test_interval - 1):

            dark_model.eval()
            pred_model.eval()

            start_time = timeit.default_timer()

            running_loss = 0.0
            running_cylinder_corrects = 0.0
            running_loss_cylinder_reg = 0.0
            running_loss_cylinder_recons = 0.0

            test_targ = torch.zeros(size=(1, 50, 4))
            test_pred = torch.zeros(size=(1, 50, 4))

            test_recons_targ = torch.zeros(size=(1, 100, 4))
            test_recons_pred = torch.zeros(size=(1, 100, 4))

            preds = []
            trues = []

            for total_info in tqdm(test_dataloader):

                with torch.no_grad():
                    dark_data = total_info[0]
                    dark_targ = total_info[1]
                    dark_recons_tg = total_info[2]
                    dark_label = total_info[3]

                    dim14_data = dark_data.to(device)
                    dim14_targ = dark_targ.to(device)
                    dark_recons_tg = dark_recons_tg.to(device)

                    cylinder_array_lab = dark_label.long()
                    cylinder_array_lab = Variable(cylinder_array_lab, requires_grad=False).to(device)

                    # decoder input
                    if Configs_pred_autoformer.padding == 0:
                        dec_inp = torch.zeros(
                            [dim14_targ.shape[0], Configs_pred_autoformer.pred_len, dim14_targ.shape[-1]]).float().to(
                            device)
                    elif Configs_pred_autoformer.padding == 1:
                        dec_inp = torch.ones(
                            [dim14_targ.shape[0], Configs_pred_autoformer.pred_len, dim14_targ.shape[-1]]).float().to(
                            device)

                    dec_inp = torch.cat([dim14_data[:, :Configs_pred_autoformer.label_len, :], dec_inp],
                                        dim=1).float().to(device)

                    pred_feat, recons_feat = dark_model(dim14_data, dim14_targ)

                    output, recons_output, cylinder_sig_type = pred_model(pred_feat, recons_feat)

                    loss_reg = criterion(output, dim14_targ)
                    loss_recons = criterion(recons_output, dark_recons_tg)
                    loss_cylinder = cylinder_loss(cylinder_sig_type, cylinder_array_lab)

                    cylinder_corrects, predsc, labsc = classification_results(cylinder_sig_type, cylinder_array_lab)
                    loss_tot, param = mt_loss(loss_reg, loss_recons, loss_cylinder)

                    running_loss += loss_tot.item()*dim14_data.size(0)
                    running_cylinder_corrects += torch.mean(cylinder_corrects)
                    running_loss_cylinder_reg += loss_reg.item() * dark_data.size(0)
                    running_loss_cylinder_recons += loss_recons.item() * dark_data.size(0)

                    test_pred = torch.cat((test_pred, output.cpu()),dim=0)
                    test_targ = torch.cat((test_targ, dim14_targ.cpu()), dim=0)

                    test_recons_targ = torch.cat((test_recons_targ, recons_output.cpu()),dim=0)
                    test_recons_pred = torch.cat((test_recons_pred, dark_recons_tg.cpu()), dim=0)

            epoch_loss = running_loss / test_size
            epoch_loss_cylinder_reg = running_loss_cylinder_reg / test_size
            epoch_acc_cylinder = running_cylinder_corrects / test_size
            epoch_loss_cylinder_recons = running_loss_cylinder_recons / test_size

            print('Testing epoch_loss', epoch_loss_cylinder_reg, 'testing epoch_acc_cylinder', epoch_acc_cylinder, 'epoch_loss_cylinder_recons', epoch_loss_cylinder_recons)

            if epoch_loss_cylinder_reg < min_test_reg:
                min_test_reg = epoch_loss_cylinder_reg
                min_test_recons = epoch_loss_cylinder_recons
                max_test_cla = epoch_acc_cylinder
                epoch_store = epoch

                test_pred_store = test_pred[1:,:].numpy()
                test_targ_store = test_targ[1:,:].numpy()

                test_recons_pred_store = test_recons_pred[1:,:].numpy()
                test_recons_targ_store = test_recons_targ[1:,:].numpy()

            writer.add_scalar('data/testing_epoch_loss', epoch_loss, epoch)
            writer.add_scalar('data/testing_regression_loss', epoch_loss_cylinder_reg, epoch)
            writer.add_scalar('data/testing_classification_accuracy', epoch_acc_cylinder, epoch)
            writer.add_scalar('data/testing_reconstruction_loss', epoch_loss_cylinder_recons, epoch)

            stop_time = timeit.default_timer()
            print("Execution time Test: " + str(stop_time - start_time) + "\n")

            print('Minimum testing loss:', min_test_reg, 'maximum testing acc classification:', max_test_cla,
                  'min_test_recons', min_test_recons,'epoch_store:', epoch_store)


    print('test_targ.size',test_targ_store.shape, 'test_pred.size',test_pred_store.shape)

    plt.figure(1)
    for i in range (9):
        plt.subplot(3, 3, i+1)
        plt.plot(test_pred_store[(i+1)*120, :, 1], 'b')
        plt.plot(test_targ_store[(i+1)*120, :, 1], 'r')
        plt.legend(['pred', 'targ'])
        plt.grid()
    plt.show()

    plt.figure(2)
    for i in range (9):
        plt.subplot(3, 3, i+1)
        plt.plot(test_recons_pred_store[(i+1)*120, :, 1], 'b')
        plt.plot(test_recons_targ_store[(i+1)*120, :, 1], 'r')
        plt.legend(['recons', 'targ'])
        plt.grid()
    plt.show()

    writer.close()


if __name__ == "__main__":

    train_model(modelType=modelType, save_dir=save_dir, num_classes=num_classes, num_fault=num_fault,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval)