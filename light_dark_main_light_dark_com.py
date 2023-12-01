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
from dataset_light_dark_new_single import vehicle_state_dataset
from Networks.Transformer_dark_light_com import TF_FD_14dim
from Networks.Transformer_dark import dark_net
from Networks.Generators import dark_gen_pred, dark_gen_recons
from Networks.MultiTaskLossWrapper import cylinder_loss, MultiTaskLossWrapper_2
from utils_f import classification_results, metric
from scipy.io import savemat

# Use GPU if available else revert to CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

torch.cuda.empty_cache()
print("Device being used:", device)

nEpochs = 1             # Number of epochs for training
resume_epoch = 0        # Default is 0, change if want to resume
useTest = True          # See evolution of the test set when training
nTestInterval = 1       # Run on test set every nTestInterval epochs
snapshot = nEpochs      # Store a model every snapshot epochs
lr = 1e-4               # Learning rate
# pre_h = 1

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

# save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
save_dir = os.path.join(save_dir_root, 'run', 'run_' + '0')

modelName = 'DisT_fault'
modelType = modelName + '_TF'
saveName = modelName

rand_flg = True

print('savename {}'.format(saveName))


def wrtieresults_classification(testing_pred, testing_label, ind):
    txtfile_name = '_' + ind + '_' + 'results_classification.mat'
    # stop_time = str(timeit.default_timer())
    txtfile_name = modelType + txtfile_name

    mdic = {"predict": testing_pred, "label": testing_label}
    savemat(txtfile_name, mdic)


def wrtieresults_regression(testing_pred, testing_label):
    txtfile_name = 'com_results_regression.mat'
    # stop_time = str(timeit.default_timer())
    txtfile_name = modelType + txtfile_name
    print('in writer testing_pred', testing_pred.shape)
    mdic = {"predict": testing_pred, "label": testing_label}
    savemat(txtfile_name, mdic)


# CHK_PATH = 'E:\Projects\Brake\District/run/run_0\models\Mar27_15-51-32_DPC000627_epoch-347.pth'
# CHK_PATH = 'E:\Projects\Brake\District/run/run_0\Store\Apr01_11-24-04_DPC000627_epoch-507.pth'
# CHK_PATH = 'E:\Projects\Brake\District/run/run_0\models\Apr05_20-53-36_DPC000627_epoch-502.pth'
# CHK_PATH = 'E:\Projects\Brake\District/run/run_0\models\May24_09-49-59_DPC000627_epoch-589.pth'
# CHK_PATH = 'E:\Projects\Brake\District/run/run_0\models\Jul06_14-03-34_DPC000627_epoch-489.pth'
CHK_PATH = 'E:\Projects\Brake\District/run/run_0\models\Jul31_09-26-31_DPC000742.pth'

def train_model(modelType=modelType, save_dir=save_dir, num_classes=num_classes, num_fault=num_fault,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    print('Training in progress')
    print('modelType:', modelType)

    light_model = TF_FD_14dim(num_fault=num_fault, device=device).to(device)
    checkpoint = torch.load(CHK_PATH)

    # dark_model = dark_net(device=device).to(device)
    # dark_model.load_state_dict(checkpoint['dark_model_state_dict'])

    dark_model_pred = dark_gen_pred(feat_dim=4, device=device).to(device)
    dark_model_recons = dark_gen_recons(feat_dim=4, device=device).to(device)
    dark_model_pred.load_state_dict(checkpoint['dark_model_pred_state_dict'])
    dark_model_recons.load_state_dict(checkpoint['dark_model_recons_state_dict'])

    mt_loss = MultiTaskLossWrapper_2(task_num=2)

    train_params = [
        {'params': light_model.parameters(), 'lr': lr},
        {'params': mt_loss.parameters(), 'lr': lr},
    ]

    print('train_params', train_params)

    print('Total params: %.2fM' % (sum(p.numel() for p in light_model.parameters()) * 4 / 1000000.0))

    criterion = nn.MSELoss()

    optimizer = optim.Adam(train_params, lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=350,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU

        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        # model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    print('tensorboard log_dir', log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    bat_size = 64

    train_dataset = vehicle_state_dataset(split='train')
    test_dataset = vehicle_state_dataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=bat_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bat_size, shuffle=False, num_workers=4, pin_memory=True)

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)

    print('test_size', test_size)

    min_test_reg=10
    max_test_cla=0.5
    epoch_store = 0


    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        print('epoch {}'.format(epoch))
        start_time = timeit.default_timer()
        print('start time', start_time)

        optimizer.step()
        scheduler.step()

        light_model.train()

        # reset the running loss and corrects
        running_loss = 0.0
        running_cylinder_corrects = 0.0
        running_loss_cylinder_reg = 0.0

        for total_info in tqdm(train_dataloader):

            light_data = total_info[0]
            light_targ = total_info[1]
            light_label = total_info[2]

            dim14_data = light_data.to(device)
            dim14_targ = light_targ.to(device)

            cylinder_array_lab = light_label.long()
            cylinder_array_lab = Variable(cylinder_array_lab, requires_grad=False).to(device)
            # print('cylinder_array_lab.size', cylinder_array_lab.size(), 'cylinder_array_lab[3].size:', cylinder_array_lab[3].size())

            # pred_feat, recons_feat = dark_model(dim14_data, dim14_targ)
            recons_feat = dark_model_recons(dim14_data)
            pred_feat = dark_model_pred(dim14_data, dim14_targ, recons_feat)

            output, cylinder_sig_type = light_model(dim14_data, dim14_targ, pred_feat, recons_feat)

            loss_reg = criterion(output, dim14_targ)

            loss_cylinder = cylinder_loss(cylinder_sig_type, cylinder_array_lab)

            cylinder_corrects, predsc, labsc = classification_results(cylinder_sig_type, cylinder_array_lab)

            # print('training cylinder_corrects', cylinder_corrects, 'training cylinder_corrects/batch', cylinder_corrects / bat_size)
            # print('training loss_cylinder_reg', loss_reg)

            loss_tot, param = mt_loss(loss_reg, loss_cylinder)

            optimizer.zero_grad()
            loss_tot.backward()
            # loss_tot.backward()
            optimizer.step()

            running_loss += loss_tot.item() * dim14_data.size(0)
            running_cylinder_corrects += torch.mean(cylinder_corrects)
            running_loss_cylinder_reg += loss_reg.item() * light_data.size(0)

        epoch_loss = running_loss / train_size
        epoch_loss_cylinder_reg = running_loss_cylinder_reg / train_size
        epoch_acc_cylinder = running_cylinder_corrects / train_size

        print('Training epoch_loss', epoch_loss_cylinder_reg, 'running_cylinder_corrects', epoch_acc_cylinder)

        writer.add_scalar('data/training_epoch_loss', epoch_loss, epoch)
        writer.add_scalar('data/training_regression_loss', epoch_loss_cylinder_reg, epoch)
        writer.add_scalar('data/training_classification_accuracy', epoch_acc_cylinder, epoch)

        stop_time = timeit.default_timer()
        print("Execution time train: " + str(stop_time - start_time) + "\n")

        if useTest and epoch % test_interval == (test_interval - 1):

            light_model.eval()

            start_time = timeit.default_timer()

            running_loss = 0.0
            running_cylinder_corrects = 0.0
            running_loss_cylinder_reg = 0.0

            test_targ = torch.zeros(size=(1, 50, 4))
            test_pred = torch.zeros(size=(1, 50, 4))

            preds = []
            trues = []

            running_pred_intent_c1 = []
            running_pred_label_c1 = []
            running_pred_intent_c2 = []
            running_pred_label_c2 = []
            running_pred_intent_c3 = []
            running_pred_label_c3 = []
            running_pred_intent_c4 = []
            running_pred_label_c4 = []

            for total_info in tqdm(test_dataloader):
                with torch.no_grad():
                    light_data = total_info[0]
                    light_targ = total_info[1]
                    light_label = total_info[2]

                    dim14_data = light_data.to(device)
                    dim14_targ = light_targ.to(device)

                    cylinder_array_lab = light_label.long()
                    cylinder_array_lab = Variable(cylinder_array_lab, requires_grad=False).to(device)
                    # print('cylinder_array_lab.size', cylinder_array_lab.size(), 'cylinder_array_lab[3].size:', cylinder_array_lab[3].size())

                    # pred_feat, recons_feat = dark_model(dim14_data, dim14_targ)

                    recons_feat = dark_model_recons(dim14_data)
                    pred_feat = dark_model_pred(dim14_data, dim14_targ, recons_feat)

                    output, cylinder_sig_type = light_model(dim14_data, dim14_targ, pred_feat, recons_feat)

                    loss_reg = criterion(output, dim14_targ)
                    cylinder_corrects, predsc, labsc = classification_results(cylinder_sig_type, cylinder_array_lab)
                    # print('testing cylinder_corrects', cylinder_corrects, 'testing cylinder_corrects/batch', cylinder_corrects / bat_size)
                    # print('testing loss_cylinder_reg', loss_reg)

                    loss_cylinder = cylinder_loss(cylinder_sig_type, cylinder_array_lab)
                    loss_tot, param = mt_loss(loss_reg, loss_cylinder)

                    # print('prediction loss:', loss)

                    running_loss += loss_tot.item()*dim14_data.size(0)
                    running_cylinder_corrects += torch.mean(cylinder_corrects)
                    running_loss_cylinder_reg += loss_reg.item() * light_data.size(0)

                    preds_tmp_intent_c1 = predsc[0].clone().detach().cpu().numpy()
                    lab_tmp_intent_c1 = labsc[0].clone().detach().cpu().numpy()
                    running_pred_intent_c1 = np.append(running_pred_intent_c1, preds_tmp_intent_c1)
                    running_pred_label_c1 = np.append(running_pred_label_c1, lab_tmp_intent_c1)

                    preds_tmp_intent_c2 = predsc[1].clone().detach().cpu().numpy()
                    lab_tmp_intent_c2 = labsc[1].clone().detach().cpu().numpy()
                    running_pred_intent_c2 = np.append(running_pred_intent_c2, preds_tmp_intent_c2)
                    running_pred_label_c2 = np.append(running_pred_label_c2, lab_tmp_intent_c2)

                    preds_tmp_intent_c3 = predsc[2].clone().detach().cpu().numpy()
                    lab_tmp_intent_c3 = labsc[2].clone().detach().cpu().numpy()
                    running_pred_intent_c3 = np.append(running_pred_intent_c3, preds_tmp_intent_c3)
                    running_pred_label_c3 = np.append(running_pred_label_c3, lab_tmp_intent_c3)

                    preds_tmp_intent_c4 = predsc[3].clone().detach().cpu().numpy()
                    lab_tmp_intent_c4 = labsc[3].clone().detach().cpu().numpy()
                    running_pred_intent_c4 = np.append(running_pred_intent_c4, preds_tmp_intent_c4)
                    running_pred_label_c4 = np.append(running_pred_label_c4, lab_tmp_intent_c4)

                    test_pred = torch.cat((test_pred, output.cpu()),dim=0)
                    test_targ = torch.cat((test_targ, dim14_targ.cpu()), dim=0)

            stop_time = timeit.default_timer()
            print("Execution time Test: " + str(stop_time - start_time) + "\n")

            epoch_loss = running_loss / test_size
            epoch_loss_cylinder_reg = running_loss_cylinder_reg / test_size
            epoch_acc_cylinder = running_cylinder_corrects / test_size

            print('Testing epoch_loss', epoch_loss_cylinder_reg, 'testing epoch_acc_cylinder', epoch_acc_cylinder)

            writer.add_scalar('data/testing_epoch_loss', epoch_loss, epoch)
            writer.add_scalar('data/testing_regression_loss', epoch_loss_cylinder_reg, epoch)
            writer.add_scalar('data/testing_classification_accuracy', epoch_acc_cylinder, epoch)

            print('Minimum testing loss:', min_test_reg, 'maximum testing acc classification:', max_test_cla,'epoch_store:', epoch_store)


    plt.figure(3)
    for i in range (9):
        plt.subplot(3, 3, i+1)
        plt.plot(test_pred[(i+1)*100, :, 1], 'b')
        plt.plot(test_targ[(i+1)*100, :, 1], 'r')
        plt.legend(['pred', 'targ'])
        plt.grid()
    plt.show()

    writer.close()


if __name__ == "__main__":

    train_model(modelType=modelType, save_dir=save_dir, num_classes=num_classes, num_fault=num_fault,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval)
