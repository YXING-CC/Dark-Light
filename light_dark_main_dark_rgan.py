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
from Networks.Generators import pred_head, dark_gen_pred_lstm, dark_gen_recons_lstm, dark_gen_pred, dark_gen_recons, \
    dark_gen_pred_Film, dark_gen_recons_Film, dark_gen_pred_autoformer, dark_gen_recons_autoformer, dark_gen_pred_tcn, dark_gen_recons_tcn
from Networks.MultiTaskLossWrapper import cylinder_loss, MultiTaskLossWrapper_3, MultiTaskLossWrapper_2
from Networks.Discriminator import Conv1d_tq_discriminator
from utils_f import classification_results, metric
from Film_configs import Configs_pred, Configs_recons, Configs_pred_autoformer, Configs_recons_autoformer
from scipy.io import savemat

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Device being used:", device)
CUDA_LAUNCH_BLOCKING=1

nEpochs = 300           # Number of epochs for training
resume_epoch = 0        # Default is 0, change if want to resume
useTest = True          # See evolution of the test set when training
nTestInterval = 1       # Run on test set every nTestInterval epochs
snapshot = 50           # Store a model every snapshot epochs
lr = 8e-5               # Learning rate  3e-4 informer 5e-8 transformer
# pre_h = 1
GAN_MODE = 0

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
    txtfile_name = '_dark_results_regression.mat'
    # stop_time = str(timeit.default_timer())
    txtfile_name = modelType + txtfile_name
    print('in writer testing_pred', testing_pred.shape)
    mdic = {"predict": testing_pred, "label": testing_label}
    savemat(txtfile_name, mdic)

def train_model(modelType=modelType, save_dir=save_dir, num_classes=num_classes, num_fault=num_fault,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    print('Training in progress')
    print('modelType:', modelType)

    dis_dim = 4

    model_type = 'Transformer'

    if model_type == 'autoformer':
        netG_pred = dark_gen_pred_autoformer(Configs_pred_autoformer).to(device)
        netG_recons = dark_gen_recons_autoformer(Configs_recons_autoformer).to(device)
    elif model_type == 'film':
        netG_pred = dark_gen_pred_Film(Configs_pred).to(device)
        netG_recons = dark_gen_recons_Film(Configs_recons).to(device)
    else:
        netG_pred = dark_gen_pred_tcn(feat_dim=4, device=device).to(device)
        netG_recons = dark_gen_recons_tcn(feat_dim=4, device=device).to(device)


    pred_model = pred_head(num_fault=num_fault).to(device)

    netD_pred = Conv1d_tq_discriminator(num_layers=3, input_dim=dis_dim, output_dim=dis_dim * 16, pred_len=50).to(device)
    netD_recons = Conv1d_tq_discriminator(num_layers=3, input_dim=dis_dim, output_dim=dis_dim * 16, pred_len=100).to(device)

    mt_loss_pred = MultiTaskLossWrapper_3(task_num=3).to(device)
    mt_loss_recons = MultiTaskLossWrapper_2(task_num=2).to(device)
    mt_loss = MultiTaskLossWrapper_2(task_num=2).to(device)

    train_params = [
        {'params': pred_model.parameters(), 'lr': lr},
        {'params': mt_loss.parameters(), 'lr': lr},
        {'params': mt_loss_pred.parameters(), 'lr': lr},
        {'params': mt_loss_recons.parameters(), 'lr': lr}
    ]

    print('train_params', train_params)

    print('Total params: %.2fM' % (sum(p.numel() for p in netG_pred.parameters()) * 4 / 1000000.0))

    criterion = nn.MSELoss()

    if GAN_MODE == 0:
        criterion1 = nn.BCEWithLogitsLoss()
    else:
        criterion1 = nn.BCELoss()

    beta1 = 0.5

    optimizer_G_pred = optim.Adam(netG_pred.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=5e-4)
    optimizer_D_pred = optim.Adam(netD_pred.parameters(), lr=lr*1, betas=(beta1, 0.999), weight_decay=5e-4)
    optimizer_G_recons = optim.Adam(netG_recons.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=5e-4)
    optimizer_D_recons = optim.Adam(netD_recons.parameters(), lr=lr*1, betas=(beta1, 0.999), weight_decay=5e-4)
    optimizer_MT = optim.Adam(train_params, lr=lr*10, betas=(beta1, 0.999), weight_decay=5e-4)

    scheduler_G_recons = optim.lr_scheduler.StepLR(optimizer_G_recons, step_size=450, gamma=0.5)  # the scheduler divides the lr by 10 every 10 epochs
    scheduler_D_recons = optim.lr_scheduler.StepLR(optimizer_D_recons, step_size=450, gamma=0.5)  # the scheduler divides the lr by 10 every 10 epochs
    scheduler_G_pred = optim.lr_scheduler.StepLR(optimizer_G_pred, step_size=450, gamma=0.5)  # the scheduler divides the lr by 10 every 10 epochs
    scheduler_D_pred = optim.lr_scheduler.StepLR(optimizer_D_pred, step_size=450, gamma=0.5)  # the scheduler divides the lr by 10 every 10 epochs
    scheduler_MT = optim.lr_scheduler.StepLR(optimizer_MT, step_size=450, gamma=0.5)

    model = [pred_model, netG_pred, netD_pred, mt_loss_pred, mt_loss_recons, netG_recons, netD_recons, mt_loss]
    optimizer_GD = [optimizer_G_recons, optimizer_D_recons, optimizer_G_pred, optimizer_D_pred, optimizer_MT]
    scheduler_GD = [scheduler_G_recons, scheduler_D_recons, scheduler_G_pred, scheduler_D_pred, scheduler_MT]

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
    test_dataloader = DataLoader(test_dataset, batch_size=bat_size, shuffle=False, num_workers=4, pin_memory=True)

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)

    min_test_reg = 5.0
    max_test_cla = 0.1
    min_test_recons = 1.0

    real_label = 0.9
    fake_label = 0.0

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        print('Train epoch {}'.format(epoch))
        for i in range(len(optimizer_GD)):
            optimizer_GD[i].step()
        for i in range(len(scheduler_GD)):
            scheduler_GD[i].step()
        for i in range(len(model)):
            model[i].train()

        start_time = timeit.default_timer()
        print('start time', start_time)

        # reset the running loss and corrects
        running_loss = 0.0

        running_cylinder_corrects = 0.0
        running_loss_cylinder_reg = 0.0
        running_loss_cylinder_recons = 0.0

        running_G_loss_pred = 0.0
        running_D_loss_pred = 0.0

        running_G_loss_recons = 0.0
        running_D_loss_recons = 0.0

        running_recons_long_tq_false_to_real = 0.0
        running_recons_long_tq_real = 0.0
        running_recons_long_tq_false = 0.0

        running_pred_long_tq_false_to_real = 0.0
        running_pred_long_tq_real = 0.0
        running_pred_long_tq_false = 0.0

        running_corrects_long_false_to_real_pred = 0.0
        running_corrects_long_false_to_real_recons = 0.0

        running_D_loss_pred_real = 0.0
        running_D_loss_pred_false = 0.0

        running_D_loss_recons_real = 0.0
        running_D_loss_recons_false = 0.0

        running_corrects_long_pred_real = 0.0
        running_corrects_long_pred_false = 0.0

        running_corrects_long_recons_real = 0.0
        running_corrects_long_recons_false = 0.0

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
            # print('cylinder_array_lab.size', cylinder_array_lab.size(), 'cylinder_array_lab[3].size:', cylinder_array_lab[3].size())

            b_size = dim14_data.size()[0]

            ##################################
            # (1) forward process
            ##################################
            sm = nn.Sigmoid()
            if model_type == 'autoformer':

                # decoder input
                if Configs_pred_autoformer.padding == 0:
                    dec_inp = torch.zeros([dim14_targ.shape[0], Configs_pred_autoformer.pred_len, dim14_targ.shape[-1]]).float().to(device)
                elif Configs_pred_autoformer.padding == 1:
                    dec_inp = torch.ones([dim14_targ.shape[0], Configs_pred_autoformer.pred_len, dim14_targ.shape[-1]]).float().to(device)

                dec_inp = torch.cat([dim14_data[:, :Configs_pred_autoformer.label_len, :], dec_inp], dim=1).float().to(device)

                recons_feat = netG_recons(dim14_data, dim14_data, dim14_data, dim14_data)
                pred_feat = netG_pred(dim14_data, dim14_data, dec_inp, dec_inp)
            else:
                recons_feat = netG_recons(dim14_data)
                pred_feat = netG_pred(dim14_data, dim14_targ, recons_feat)

            output, recons_output, cylinder_sig_type = pred_model(pred_feat, recons_feat)

            ####################################
            # (2) update netG_pred and netG_recons
            ####################################
            optimizer_G_pred.zero_grad()
            optimizer_G_recons.zero_grad()
            optimizer_MT.zero_grad()

            if GAN_MODE == 0:
                label_generator = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)
            else:
                label_generator = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            real_pred = netD_pred(dim14_targ)
            false_to_real_pred = netD_pred(output)

            real_recons = netD_recons(dark_recons_tg)
            false_to_real_recons = netD_recons(recons_output)

            if GAN_MODE == 0:
                loss_G_pred = criterion1(false_to_real_pred - real_pred.mean(0, keepdim=True), label_generator)
                false_to_real_pred_labelize = torch.round(sm(false_to_real_pred)) * 0.9

                loss_G_recons = criterion1(false_to_real_recons - real_recons.mean(0, keepdim=True), label_generator)
                false_to_real_recons_labelize = torch.round(sm(false_to_real_recons)) * 0.9
            else:
                loss_G_TQ = criterion1(false_tq_to_real, label_generator)
                false_tq_to_real_labelize = torch.round((false_tq_to_real)) * 0.9

            loss_cylinder = cylinder_loss(cylinder_sig_type, cylinder_array_lab)
            loss_reg = criterion(output, dim14_targ)
            loss_recons = criterion(recons_output, dark_recons_tg)

            loss_G_pred_mt, varibles = mt_loss_pred(loss_G_pred, loss_reg, loss_cylinder)
            loss_G_recons_mt, varibles = mt_loss_recons(loss_G_recons, loss_recons)

            loss_G, param = mt_loss(loss_G_pred_mt, loss_G_recons_mt)
            # loss_G = loss_G_pred_mt

            cylinder_corrects, predsc, labsc = classification_results(cylinder_sig_type, cylinder_array_lab)

            loss_G.backward()

            optimizer_MT.step()
            optimizer_G_pred.step()
            optimizer_G_recons.step()

            running_G_loss_pred += loss_G_pred_mt.item() * dim14_data.size(0)
            running_G_loss_recons += loss_G_recons_mt.item() * dim14_data.size(0)

            running_cylinder_corrects += torch.mean(cylinder_corrects)
            running_loss_cylinder_reg += loss_reg.item() * dark_data.size(0)
            running_loss_cylinder_recons += loss_recons.item() * dark_data.size(0)

            running_corrects_long_false_to_real_pred += torch.sum(false_to_real_pred_labelize == label_generator.data)
            running_corrects_long_false_to_real_recons += torch.sum(false_to_real_recons_labelize == label_generator.data)

            ####################################
            # (3) update D_pred and D_recons
            ####################################
            optimizer_D_pred.zero_grad()
            optimizer_D_recons.zero_grad()

            if GAN_MODE == 0:
                label_real = torch.full((b_size,1), real_label, dtype=torch.float, device=device)
                label_false = torch.full((b_size,1), fake_label, dtype=torch.float, device=device)
            else:
                label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                label_false = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            pred_real = netD_pred(dim14_targ)
            pred_fake = netD_pred(output.detach())

            recons_real = netD_recons(dark_recons_tg)
            recons_fake = netD_recons(recons_output.detach())

            if GAN_MODE == 0:
                loss_D_real_pred = criterion1(pred_real - pred_fake.mean(0, keepdim=True), label_real)
                loss_D_fake_pred = criterion1(pred_fake - pred_real.mean(0, keepdim=True), label_false)
                pred_real_labelize = torch.round(sm(pred_real)) * 0.9
                pred_fake_labelize = torch.round(sm(pred_fake)) * 0.9

                loss_D_real_recons = criterion1(recons_real - recons_fake.mean(0, keepdim=True), label_real)
                loss_D_fake_recons = criterion1(recons_fake - recons_real.mean(0, keepdim=True), label_false)
                recons_real_labelize = torch.round(sm(recons_real)) * 0.9
                recons_fake_labelize = torch.round(sm(recons_real)) * 0.9
            else:
                loss_D_tq_real = criterion1(pred_tq_real, label_real)
                loss_D_tq_fake = criterion1(pred_tq_fake, label_false)
                pred_tq_real_labelize = torch.round((pred_tq_real)) * 0.9
                pred_tq_fake_labelize = torch.round((pred_tq_fake)) * 0.9

            loss_D_pred = (loss_D_real_pred + loss_D_fake_pred) / 2
            loss_D_recons = (loss_D_real_recons + loss_D_fake_recons) / 2

            loss_D = loss_D_pred + loss_D_recons

            loss_D.backward()

            optimizer_D_pred.step()
            optimizer_D_recons.step()

            running_D_loss_pred += loss_D_pred.item() * dim14_data.size(0)
            running_D_loss_recons += loss_D_recons.item() * dim14_data.size(0)

            running_D_loss_pred_real += loss_D_real_pred.item() * dim14_data.size(0)
            running_D_loss_pred_false += loss_D_fake_pred.item() * dim14_data.size(0)

            running_D_loss_recons_real += loss_D_real_recons.item() * dim14_data.size(0)
            running_D_loss_recons_false += loss_D_fake_recons.item() * dim14_data.size(0)

            running_corrects_long_pred_real += torch.sum(pred_real_labelize == label_real.data)
            running_corrects_long_pred_false += torch.sum(pred_fake_labelize == label_false.data)

            running_corrects_long_recons_real += torch.sum(recons_real_labelize == label_real.data)
            running_corrects_long_recons_false += torch.sum(recons_fake_labelize == label_false.data)

        epoch_G_loss_pred = running_G_loss_pred / train_size
        epoch_G_loss_recons = running_G_loss_recons / train_size

        epoch_D_loss_pred = running_D_loss_pred / train_size
        epoch_D_loss_recons = running_D_loss_recons / train_size

        epoch_loss_cylinder_reg = running_loss_cylinder_reg / train_size
        epoch_loss_cylinder_recons = running_loss_cylinder_recons / train_size
        epoch_acc_cylinder = running_cylinder_corrects / train_size

        epoch_D_loss_pred_real = running_D_loss_pred_real / train_size
        epoch_D_loss_pred_fake = running_D_loss_pred_false / train_size

        epoch_D_loss_recons_real = running_D_loss_recons_real / train_size
        epoch_D_loss_recons_fake = running_D_loss_recons_false / train_size

        epoch_acc_pred_false_to_real = running_corrects_long_false_to_real_pred / train_size
        epoch_acc_recons_false_to_real = running_corrects_long_false_to_real_recons / train_size

        epoch_corrects_pred_real = running_corrects_long_pred_real / train_size
        epoch_corrects_pred_fake = running_corrects_long_pred_false / train_size

        epoch_corrects_recons_real = running_corrects_long_recons_real / train_size
        epoch_corrects_recons_fake = running_corrects_long_recons_false / train_size

        print('Training epoch_loss', epoch_loss_cylinder_reg, 'running_cylinder_corrects', epoch_acc_cylinder, 'loss_cylinder_recons', epoch_loss_cylinder_recons)

        writer.add_scalar('data/training_epoch_G_loss_pred', epoch_G_loss_pred, epoch)
        writer.add_scalar('data/training_epoch_G_loss_recons', epoch_G_loss_recons, epoch)
        writer.add_scalar('data/training_epoch_D_loss_pred', epoch_D_loss_pred, epoch)
        writer.add_scalar('data/training_epoch_D_loss_recons', epoch_D_loss_recons, epoch)
        writer.add_scalar('data/training_epoch_D_loss_pred_real', epoch_D_loss_pred_real, epoch)
        writer.add_scalar('data/training_epoch_D_loss_pred_fake', epoch_D_loss_pred_fake, epoch)
        writer.add_scalar('data/training_epoch_D_loss_recons_real', epoch_D_loss_recons_real, epoch)
        writer.add_scalar('data/training_epoch_D_loss_recons_fake', epoch_D_loss_recons_fake, epoch)

        writer.add_scalar('data/training_epoch_acc_pred_false_to_real', epoch_acc_pred_false_to_real, epoch)
        writer.add_scalar('data/training_epoch_acc_recons_false_to_real', epoch_acc_recons_false_to_real, epoch)
        writer.add_scalar('data/training_epoch_corrects_pred_real', epoch_corrects_pred_real, epoch)
        writer.add_scalar('data/training_epoch_corrects_pred_fake', epoch_corrects_pred_fake, epoch)
        writer.add_scalar('data/training_epoch_corrects_recons_real', epoch_corrects_recons_real, epoch)
        writer.add_scalar('data/training_epoch_corrects_recons_fake', epoch_corrects_recons_fake, epoch)

        writer.add_scalar('data/training_regression_loss', epoch_loss_cylinder_reg, epoch)
        writer.add_scalar('data/training_classification_accuracy', epoch_acc_cylinder, epoch)
        writer.add_scalar('data/training_reconstruction_loss', epoch_loss_cylinder_recons, epoch)

        stop_time = timeit.default_timer()
        print("Execution time train: " + str(stop_time - start_time) + "\n")

        if useTest and epoch % test_interval == (test_interval - 1):

            for i in range(len(model)):
                model[i].eval()

            start_time = timeit.default_timer()

            running_cylinder_corrects = 0.0
            running_loss_cylinder_reg = 0.0
            running_loss_cylinder_recons = 0.0

            running_G_loss_pred = 0.0
            running_D_loss_pred = 0.0

            running_G_loss_recons = 0.0
            running_D_loss_recons = 0.0

            running_recons_long_tq_false_to_real = 0.0
            running_recons_long_tq_real = 0.0
            running_recons_long_tq_false = 0.0

            running_pred_long_tq_false_to_real = 0.0
            running_pred_long_tq_real = 0.0
            running_pred_long_tq_false = 0.0

            running_corrects_long_false_to_real_pred = 0.0
            running_corrects_long_false_to_real_recons = 0.0

            running_D_loss_pred_real = 0.0
            running_D_loss_pred_false = 0.0

            running_D_loss_recons_real = 0.0
            running_D_loss_recons_false = 0.0

            running_corrects_long_pred_real = 0.0
            running_corrects_long_pred_false = 0.0

            running_corrects_long_recons_real = 0.0
            running_corrects_long_recons_false = 0.0

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

                    b_size = dim14_data.size()[0]
                    ##################################
                    # (1) forward process
                    ##################################
                    sm = nn.Sigmoid()

                    if model_type == 'autoformer':

                        # decoder input
                        if Configs_pred_autoformer.padding == 0:
                            dec_inp = torch.zeros([dim14_targ.shape[0], Configs_pred_autoformer.pred_len,
                                                   dim14_targ.shape[-1]]).float().to(device)
                        elif Configs_pred_autoformer.padding == 1:
                            dec_inp = torch.ones([dim14_targ.shape[0], Configs_pred_autoformer.pred_len,
                                                  dim14_targ.shape[-1]]).float().to(device)

                        dec_inp = torch.cat([dim14_data[:, :Configs_pred_autoformer.label_len, :], dec_inp],
                                            dim=1).float().to(device)

                        recons_feat = netG_recons(dim14_data, dim14_data, dim14_data, dim14_data)
                        pred_feat = netG_pred(dim14_data, dim14_data, dec_inp, dec_inp)

                    else:
                        recons_feat = netG_recons(dim14_data)
                        pred_feat = netG_pred(dim14_data, dim14_targ, recons_feat)

                    output, recons_output, cylinder_sig_type = pred_model(pred_feat, recons_feat)

                    ####################################
                    # (2) update netG_pred and netG_recons
                    ####################################
                    optimizer_G_pred.zero_grad()
                    optimizer_G_recons.zero_grad()
                    optimizer_MT.zero_grad()

                    if GAN_MODE == 0:
                        label_generator = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)
                    else:
                        label_generator = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                    real_pred = netD_pred(dim14_targ)
                    false_to_real_pred = netD_pred(output)

                    real_recons = netD_recons(dark_recons_tg)
                    false_to_real_recons = netD_recons(recons_output)

                    if GAN_MODE == 0:
                        loss_G_pred = criterion1(false_to_real_pred - real_pred.mean(0, keepdim=True), label_generator)
                        false_to_real_pred_labelize = torch.round(sm(false_to_real_pred)) * 0.9

                        loss_G_recons = criterion1(false_to_real_recons - real_recons.mean(0, keepdim=True),
                                                   label_generator)
                        false_to_real_recons_labelize = torch.round(sm(false_to_real_recons)) * 0.9
                    else:
                        loss_G_TQ = criterion1(false_tq_to_real, label_generator)
                        false_tq_to_real_labelize = torch.round((false_tq_to_real)) * 0.9

                    loss_cylinder = cylinder_loss(cylinder_sig_type, cylinder_array_lab)
                    loss_reg = criterion(output, dim14_targ)
                    loss_recons = criterion(recons_output, dark_recons_tg)

                    loss_G_pred, varibles = mt_loss_pred(loss_G_pred, loss_reg, loss_cylinder)
                    loss_G_recons, varibles = mt_loss_recons(loss_G_recons, loss_recons)

                    cylinder_corrects, predsc, labsc = classification_results(cylinder_sig_type, cylinder_array_lab)

                    running_G_loss_pred += loss_G_pred.item() * dim14_data.size(0)
                    running_G_loss_recons += loss_G_recons.item() * dim14_data.size(0)

                    running_cylinder_corrects += torch.mean(cylinder_corrects)
                    running_loss_cylinder_reg += loss_reg.item() * dark_data.size(0)
                    running_loss_cylinder_recons += loss_recons.item() * dark_data.size(0)

                    running_corrects_long_false_to_real_pred += torch.sum(
                        false_to_real_pred_labelize == label_generator.data)
                    running_corrects_long_false_to_real_recons += torch.sum(
                        false_to_real_recons_labelize == label_generator.data)

                    ####################################
                    # (3) update D_pred and D_recons
                    ####################################
                    optimizer_D_pred.zero_grad()
                    optimizer_D_recons.zero_grad()

                    if GAN_MODE == 0:
                        label_real = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)
                        label_false = torch.full((b_size, 1), fake_label, dtype=torch.float, device=device)
                    else:
                        label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                        label_false = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

                    pred_real = netD_pred(dim14_targ)
                    pred_fake = netD_pred(output)

                    recons_real = netD_recons(dark_recons_tg)
                    recons_fake = netD_recons(recons_output)

                    if GAN_MODE == 0:
                        loss_D_real_pred = criterion1(pred_real - pred_fake.mean(0, keepdim=True), label_real)
                        loss_D_fake_pred = criterion1(pred_fake - pred_real.mean(0, keepdim=True), label_false)
                        pred_real_labelize = torch.round(sm(pred_real)) * 0.9
                        pred_fake_labelize = torch.round(sm(pred_fake)) * 0.9

                        loss_D_real_recons = criterion1(recons_real - recons_fake.mean(0, keepdim=True), label_real)
                        loss_D_fake_recons = criterion1(recons_fake - recons_real.mean(0, keepdim=True), label_false)
                        recons_real_labelize = torch.round(sm(recons_real)) * 0.9
                        recons_fake_labelize = torch.round(sm(recons_real)) * 0.9
                    else:
                        loss_D_tq_real = criterion1(pred_tq_real, label_real)
                        loss_D_tq_fake = criterion1(pred_tq_fake, label_false)
                        pred_tq_real_labelize = torch.round((pred_tq_real)) * 0.9
                        pred_tq_fake_labelize = torch.round((pred_tq_fake)) * 0.9

                    loss_D_pred = (loss_D_real_pred + loss_D_fake_pred) / 2
                    loss_D_recons = (loss_D_real_recons + loss_D_fake_recons) / 2

                    running_D_loss_pred += loss_D_pred.item() * dark_data.size(0)
                    running_D_loss_recons += loss_D_recons.item() * dark_data.size(0)

                    running_D_loss_pred_real += loss_D_real_pred.item() * dark_data.size(0)
                    running_D_loss_pred_false += loss_D_fake_pred.item() * dark_data.size(0)

                    running_D_loss_recons_real += loss_D_real_recons.item() * dark_data.size(0)
                    running_D_loss_recons_false += loss_D_fake_recons.item() * dark_data.size(0)

                    running_corrects_long_pred_real += torch.sum(pred_real_labelize == label_real.data)
                    running_corrects_long_pred_false += torch.sum(pred_fake_labelize == label_false.data)

                    running_corrects_long_recons_real += torch.sum(recons_real_labelize == label_real.data)
                    running_corrects_long_recons_false += torch.sum(recons_fake_labelize == label_false.data)

                    test_pred = torch.cat((test_pred, output.cpu()),dim=0)
                    test_targ = torch.cat((test_targ, dim14_targ.cpu()), dim=0)

                    test_recons_targ = torch.cat((test_recons_targ, recons_output.cpu()),dim=0)
                    test_recons_pred = torch.cat((test_recons_pred, dark_recons_tg.cpu()), dim=0)

            epoch_G_loss_pred = running_G_loss_pred / test_size
            epoch_G_loss_recons = running_G_loss_recons / test_size

            epoch_D_loss_pred = running_D_loss_pred / test_size
            epoch_D_loss_recons = running_D_loss_recons / test_size

            epoch_loss_cylinder_reg = running_loss_cylinder_reg / test_size
            epoch_loss_cylinder_recons = running_loss_cylinder_recons / test_size
            epoch_acc_cylinder = running_cylinder_corrects / test_size

            epoch_D_loss_pred_real = running_D_loss_pred_real / test_size
            epoch_D_loss_pred_fake = running_D_loss_pred_false / test_size

            epoch_D_loss_recons_real = running_D_loss_recons_real / test_size
            epoch_D_loss_recons_fake = running_D_loss_recons_false / test_size

            epoch_acc_pred_false_to_real = running_corrects_long_false_to_real_pred / test_size
            epoch_acc_recons_false_to_real = running_corrects_long_false_to_real_recons / test_size

            epoch_corrects_pred_real = running_corrects_long_pred_real / test_size
            epoch_corrects_pred_fake = running_corrects_long_pred_false / test_size

            epoch_corrects_recons_real = running_corrects_long_recons_real / test_size
            epoch_corrects_recons_fake = running_corrects_long_recons_false / test_size

            print('Testing epoch_loss', epoch_loss_cylinder_reg, 'testing epoch_acc_cylinder', epoch_acc_cylinder, 'epoch_loss_cylinder_recons', epoch_loss_cylinder_recons)

            writer.add_scalar('data/testing_epoch_G_loss_pred', epoch_G_loss_pred, epoch)
            writer.add_scalar('data/testing_epoch_G_loss_recons', epoch_G_loss_recons, epoch)
            writer.add_scalar('data/testing_epoch_D_loss_pred', epoch_D_loss_pred, epoch)
            writer.add_scalar('data/testing_epoch_D_loss_recons', epoch_D_loss_recons, epoch)
            writer.add_scalar('data/testing_epoch_D_loss_pred_real', epoch_D_loss_pred_real, epoch)
            writer.add_scalar('data/testing_epoch_D_loss_pred_fake', epoch_D_loss_pred_fake, epoch)
            writer.add_scalar('data/testing_epoch_D_loss_recons_real', epoch_D_loss_recons_real, epoch)
            writer.add_scalar('data/testing_epoch_D_loss_recons_fake', epoch_D_loss_recons_fake, epoch)

            writer.add_scalar('data/testing_epoch_acc_pred_false_to_real', epoch_acc_pred_false_to_real, epoch)
            writer.add_scalar('data/testing_epoch_acc_recons_false_to_real', epoch_acc_recons_false_to_real, epoch)
            writer.add_scalar('data/testing_epoch_corrects_pred_real', epoch_corrects_pred_real, epoch)
            writer.add_scalar('data/testing_epoch_corrects_pred_fake', epoch_corrects_pred_fake, epoch)
            writer.add_scalar('data/testing_epoch_corrects_recons_real', epoch_corrects_recons_real, epoch)
            writer.add_scalar('data/testing_epoch_corrects_recons_fake', epoch_corrects_recons_fake, epoch)

            writer.add_scalar('data/testing_regression_loss', epoch_loss_cylinder_reg, epoch)
            writer.add_scalar('data/testing_classification_accuracy', epoch_acc_cylinder, epoch)
            writer.add_scalar('data/testing_reconstruction_loss', epoch_loss_cylinder_recons, epoch)

            if epoch_loss_cylinder_reg < min_test_reg:
                min_test_reg = epoch_loss_cylinder_reg
                min_test_recons = epoch_loss_cylinder_recons
                max_test_cla = epoch_acc_cylinder
                epoch_store = epoch

                test_pred_store = test_pred[1:,:,:].numpy()
                test_targ_store = test_targ[1:,:,:].numpy()

                test_recons_pred_store = test_recons_pred[1:,:,:].numpy()
                test_recons_targ_store = test_recons_targ[1:,:,:].numpy()

                if epoch > 300:
                    torch.save({
                        'epoch': epoch + 1,
                        'dark_model_pred_state_dict': netG_pred.state_dict(),
                        'dark_model_recons_state_dict': netG_recons.state_dict(),
                        'optimizer_G_pred': optimizer_G_pred.state_dict(),
                        'optimizer_G_recons': optimizer_G_recons.state_dict(),
                    }, os.path.join(log_dir + '.pth'))
                    print("Save model at {}\n".format(os.path.join(log_dir + '_epoch-' + str(epoch) + '.pth')))
                    wrtieresults_regression(test_recons_pred_store, test_recons_targ_store)


            stop_time = timeit.default_timer()
            print("Execution time Test: " + str(stop_time - start_time) + "\n")

            print('Minimum testing pred loss:', min_test_reg, 'Minimum testing recons loss:',
                  min_test_recons,'maximum testing acc classification:', max_test_cla, 'epoch_store:', epoch_store)

    print('test_targ.size',test_targ_store.shape, 'test_pred.size',test_pred_store.shape)

    ext = 15
    plt.figure(1)
    for i in range (9):
        plt.subplot(3, 3, i+1)
        plt.plot(test_pred_store[(i+1)*120+ext, :, 1], 'b')
        plt.plot(test_targ_store[(i+1)*120+ext, :, 1], 'r')
        plt.legend(['pred', 'targ'])
        plt.grid()
    plt.show()

    plt.figure(2)
    for i in range (9):
        plt.subplot(3, 3, i+1)
        plt.plot(test_recons_pred_store[(i+1)*120+ext, :, 1], 'b')
        plt.plot(test_recons_targ_store[(i+1)*120+ext, :, 1], 'r')
        plt.legend(['recons', 'targ'])
        plt.grid()
    plt.show()

    writer.close()

if __name__ == "__main__":

    train_model(modelType=modelType, save_dir=save_dir, num_classes=num_classes, num_fault=num_fault,  lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval)