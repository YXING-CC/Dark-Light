clc;clear

load data.mat

ind = [3 4 5 6 48 49 53 54 56 58 59 60 64 65];

% [lf_pressure (3-1), rf_pressure (4-2), lr_pressure (5-3), rr_presure (6-4), 
%  lf_wheel_speed (48-5), rf_wheel_speed (49-6),speed (53-7), motor_target_torque (54-8), 
%  acc_pedal(56-9), batt_soc (58-10)  batt_current (59-11),batt_voltage (60-12) 
%  motor speed (64-13), motor_real_torque (65-14)]

% cylinder: lf_pressure, rf_pressure, lr_pressure, rr_presure [1,2,3,4]
% can: lf_wheel_speed, rf_wheel_speed, speed, acc_padal [5,6,7,9]
% motor: motor_target_torque, motor_speed, motor_real_torque [8,13,14]
% battery: battery_soc, battery_current, battery_voltage [10,11,12]

traindata = traindata(:,ind);
testdata = testdata(1:222222,ind);

for i = 1:length(ind)
    traindata(:,i) = smooth(traindata(:,i));
    testdata(:,i) = smooth(testdata(:,i));
end

mu = mean(traindata',2);
sig = std(traindata',0,2);
max_s = max(traindata);

trainnormal = (traindata' - mu) ./ sig;
testnormal = (testdata' - mu) ./ sig;

trainnormal = trainnormal';
testnormal = testnormal';

% save normalizeddata.mat trainnormal testnormal


%%

cylinder_ind = [1,2,3,4];
can_ind = [5,6,7,9];
motor_ind = [8,13,14];
batt_ind = [10,11,12];

%------------mu&sig--------------
cylinder_mu = mu(cylinder_ind)';
can_mu = mu(can_ind)';
motor_mu = mu(motor_ind)';
batt_mu = mu(batt_ind)';

cylinder_sig = sig(cylinder_ind)';
can_sig = sig(can_ind)';
motor_sig = sig(motor_ind)';
batt_sig = sig(batt_ind)';

cylinder_max = max_s(cylinder_ind)';
can_max = max_s(can_ind)';
motor_max = max_s(motor_ind)';
batt_max = max_s(batt_ind)';
%--------------------------------
seq_len = 150;

len_train = length(traindata);
len_test=  length(testdata);

num_train_seq = floor(len_train / seq_len);
num_test_seq = floor(len_test / seq_len);

cylid_train = cell(num_train_seq, 1);
can_train = cell(num_train_seq, 1);
motor_train = cell(num_train_seq, 1);
batt_train = cell(num_train_seq, 1);

cylid_test = cell(num_test_seq, 1);
can_test = cell(num_test_seq, 1);
motor_test = cell(num_test_seq, 1);
batt_test = cell(num_test_seq, 1);


for i = 1:num_train_seq
    cylid_train{i} = traindata((i-1)*seq_len+1:i*seq_len, cylinder_ind);
    can_train{i} = traindata((i-1)*seq_len+1:i*seq_len, can_ind);
    motor_train{i} = traindata((i-1)*seq_len+1:i*seq_len, motor_ind);
    batt_train{i} = traindata((i-1)*seq_len+1:i*seq_len, batt_ind);
end

for i = 1:num_test_seq
    cylid_test{i} = testdata((i-1)*seq_len+1:i*seq_len, cylinder_ind);
    can_test{i} = testdata((i-1)*seq_len+1:i*seq_len, can_ind);
    motor_test{i} = testdata((i-1)*seq_len+1:i*seq_len, motor_ind);
    batt_test{i} = testdata((i-1)*seq_len+1:i*seq_len, batt_ind);
end

org_train = cell(4,1);
org_test = cell(4,1);
mu_cell = cell(4,1);
sig_cell = cell(4,1);
max_cell = cell(4,1);

org_train{1} = cylid_train;
org_train{2} = can_train;
org_train{3} = motor_train;
org_train{4} = batt_train;

org_test{1} = cylid_test;
org_test{2} = can_test;
org_test{3} = motor_test;
org_test{4} = batt_test;

mu_cell{1} = cylinder_mu;
mu_cell{2} = can_mu;
mu_cell{3} = motor_mu;
mu_cell{4} = batt_mu;

sig_cell{1} = cylinder_sig;
sig_cell{2} = can_sig;
sig_cell{3} = motor_sig;
sig_cell{4} = batt_sig;

max_cell{1}= cylinder_max;
max_cell{2}= can_max;
max_cell{3}= motor_max;
max_cell{4}= batt_max;

%%
org_lable_train = cell(4,1);
org_lable_test= cell(4,1);
for i = 1:4
    data = org_train{i};
    len_d = length(data);
    cell_lab = cell(len_d,1);
    dd = data{1};
    [len_dd,dim] = size(dd);
    for j = 1:len_d
        cell_lab{j} = zeros(1,dim);
    end
    org_lable_train{i} = cell_lab;
    
    data_test = org_test{i};
    len_d = length(data_test);
    cell_lab = cell(len_d,1);
    dd = data_test{1};
    [len_dd,dim] = size(dd);
    for j = 1:len_d
        cell_lab{j} = zeros(1,dim);
    end
    org_lable_test{i} = cell_lab;
    
end

%% Generate artificial fault 

[fault_train_1, fault_lab_train_1, fault_lab_train_array_1] = generate_fault_signal(org_train,max_cell);
[fault_test_1, fault_lab_test_1, fault_lab_test_array_1] = generate_fault_signal(org_test,max_cell);

[fault_train_2, fault_lab_train_2, fault_lab_train_array_2] = generate_fault_signal(org_train,max_cell);
[fault_test_2, fault_lab_test_2, fault_lab_test_array_2] = generate_fault_signal(org_test,max_cell);

[fault_train_3, fault_lab_train_3,fault_lab_train_array_3] = generate_fault_signal(org_train,max_cell);
[fault_test_3, fault_lab_test_3, fault_lab_test_array_3] = generate_fault_signal(org_test,max_cell);

[fault_train_4, fault_lab_train_4,fault_lab_train_array_4] = generate_fault_signal(org_train,max_cell);
[fault_test_4, fault_lab_test_4, fault_lab_test_array_4] = generate_fault_signal(org_test,max_cell);


norm_fault_train_4 = generate_normalized_signal(fault_train_4, mu_cell, sig_cell);
norm_fault_test_4 = generate_normalized_signal(fault_test_4, mu_cell, sig_cell);

norm_fault_train_1 = generate_normalized_signal(fault_train_1, mu_cell, sig_cell);
norm_fault_test_1 = generate_normalized_signal(fault_test_1, mu_cell, sig_cell);

norm_fault_train_2 = generate_normalized_signal(fault_train_2, mu_cell, sig_cell);
norm_fault_test_2 = generate_normalized_signal(fault_test_2, mu_cell, sig_cell);

norm_fault_train_3 = generate_normalized_signal(fault_train_3, mu_cell, sig_cell);
norm_fault_test_3 = generate_normalized_signal(fault_test_3, mu_cell, sig_cell);

norm_org_train = generate_normalized_signal(org_train, mu_cell, sig_cell);
norm_org_test = generate_normalized_signal(org_test, mu_cell, sig_cell);


%%

[n_train,w_train] = size(norm_org_train);
[n_test, w_test] = size(norm_org_test);

cylind_train_data = cell(1,1);
cylind_train_targ = cell(1,1);
can_train_data = cell(1,1);
can_train_targ = cell(1,1);
mot_train_data = cell(1,1);
mot_train_targ = cell(1,1);
bat_train_data = cell(1,1);
bat_train_targ = cell(1,1);

cylind_test_data = cell(1,1);
cylind_test_targ = cell(1,1);
can_test_data = cell(1,1);
can_test_targ = cell(1,1);
mot_test_data = cell(1,1);
mot_test_targ = cell(1,1);
bat_test_data = cell(1,1);
bat_test_targ = cell(1,1);

data_temp = cell(4,1);

cylind_train = norm_org_train{1};
can_train = norm_org_train{2};
mot_train = norm_org_train{3};
bat_train = norm_org_train{4};

cylind_test = norm_org_test{1};
can_test = norm_org_test{2};
mot_test = norm_org_test{3};
bat_test = norm_org_test{4};

len_cyl = length(cylind_train);

for i = 1:len_cyl
    data1 = cylind_train{i};
    data2 = can_train{i};
    data3 = mot_train{i};
    data4 = bat_train{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_train_data{i,1} = data1(1:100,:);
    cylind_train_targ{i,1} = data1(101:end,:);

    can_train_data{i,1} = data2(1:100,:);
    can_train_targ{i,1} = data2(101:end,:);

    mot_train_data{i,1} = data3(1:100,:);
    mot_train_targ{i,1} = data3(101:end,:);

    bat_train_data{i,1} = data4(1:100,:);
    bat_train_targ{i,1} = data4(101:end,:);
end

len_cyl_test = length(cylind_test);

% data1 = [];
% data2 = [];
% data3 = [];
% data4 = [];

for i = 1:len_cyl_test
    data1 = cylind_test{i};
    data2 = can_test{i};
    data3 = mot_test{i};
    data4 = bat_test{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_test_data{i,1} = data1(1:100,:);
    cylind_test_targ{i,1} = data1(101:end,:);

    can_test_data{i,1} = data2(1:100,:);
    can_test_targ{i,1} = data2(101:end,:);

    mot_test_data{i,1} = data3(1:100,:);
    mot_test_targ{i,1} = data3(101:end,:);

    bat_test_data{i,1} = data4(1:100,:);
    bat_test_targ{i,1} = data4(101:end,:);
end

%% Fault 1

[n_train,w_train] = size(norm_fault_train_1);
[n_test, w_test] = size(norm_fault_test_1);

cylind_train_data_ft1 = cell(1,1);
cylind_train_targ_ft1 = cell(1,1);
can_train_data_ft1 = cell(1,1);
can_train_targ_ft1 = cell(1,1);
mot_train_data_ft1 = cell(1,1);
mot_train_targ_ft1 = cell(1,1);
bat_train_data_ft1 = cell(1,1);
bat_train_targ_ft1 = cell(1,1);

cylind_test_data_ft1 = cell(1,1);
cylind_test_targ_ft1 = cell(1,1);
can_test_data_ft1 = cell(1,1);
can_test_targ_ft1 = cell(1,1);
mot_test_data_ft1 = cell(1,1);
mot_test_targ_ft1 = cell(1,1);
bat_test_data_ft1 = cell(1,1);
bat_test_targ_ft1 = cell(1,1);

data_temp = cell(4,1);

cylind_train_ft1 = norm_fault_train_1{1};
can_train_ft1 = norm_fault_train_1{2};
mot_train_ft1 = norm_fault_train_1{3};
bat_train_ft1 = norm_fault_train_1{4};

cylind_test_ft1 = norm_fault_test_1{1};
can_test_ft1 = norm_fault_test_1{2};
mot_test_ft1 = norm_fault_test_1{3};
bat_test_ft1 = norm_fault_test_1{4};

len_cyl = length(cylind_train_ft1);

for i = 1:len_cyl
    data1 = cylind_train_ft1{i};
    data2 = can_train_ft1{i};
    data3 = mot_train_ft1{i};
    data4 = bat_train_ft1{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_train_data_ft1{i,1} = data1(1:100,:);
    cylind_train_targ_ft1{i,1} = data1(101:end,:);

    can_train_data_ft1{i,1} = data2(1:100,:);
    can_train_targ_ft1{i,1} = data2(101:end,:);

    mot_train_data_ft1{i,1} = data3(1:100,:);
    mot_train_targ_ft1{i,1} = data3(101:end,:);

    bat_train_data_ft1{i,1} = data4(1:100,:);
    bat_train_targ_ft1{i,1} = data4(101:end,:);
end

len_cyl_test = length(cylind_test_ft1);


for i = 1:len_cyl_test
    data1 = cylind_test_ft1{i};
    data2 = can_test_ft1{i};
    data3 = mot_test_ft1{i};
    data4 = bat_test_ft1{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_test_data_ft1{i,1} = data1(1:100,:);
    cylind_test_targ_ft1{i,1} = data1(101:end,:);

    can_test_data_ft1{i,1} = data2(1:100,:);
    can_test_targ_ft1{i,1} = data2(101:end,:);

    mot_test_data_ft1{i,1} = data3(1:100,:);
    mot_test_targ_ft1{i,1} = data3(101:end,:);

    bat_test_data_ft1{i,1} = data4(1:100,:);
    bat_test_targ_ft1{i,1} = data4(101:end,:);
end

%% Fault 2

[n_train,w_train] = size(norm_fault_train_2);
[n_test, w_test] = size(norm_fault_test_2);

cylind_train_data_ft2 = cell(1,1);
cylind_train_targ_ft2 = cell(1,1);
can_train_data_ft2 = cell(1,1);
can_train_targ_ft2 = cell(1,1);
mot_train_data_ft2 = cell(1,1);
mot_train_targ_ft2 = cell(1,1);
bat_train_data_ft2 = cell(1,1);
bat_train_targ_ft2 = cell(1,1);

cylind_test_data_ft2 = cell(1,1);
cylind_test_targ_ft2 = cell(1,1);
can_test_data_ft2 = cell(1,1);
can_test_targ_ft2 = cell(1,1);

mot_test_data_ft2 = cell(1,1);
mot_test_targ_ft2 = cell(1,1);
bat_test_data_ft2 = cell(1,1);
bat_test_targ_ft2 = cell(1,1);

data_temp = cell(4,1);

cylind_train_ft2 = norm_fault_train_2{1};
can_train_ft2 = norm_fault_train_2{2};
mot_train_ft2 = norm_fault_train_2{3};
bat_train_ft2 = norm_fault_train_2{4};

cylind_test_ft2 = norm_fault_test_2{1};
can_test_ft2 = norm_fault_test_2{2};
mot_test_ft2 = norm_fault_test_2{3};
bat_test_ft2 = norm_fault_test_2{4};

len_cyl = length(cylind_train_ft2);

for i = 1:len_cyl
    data1 = cylind_train_ft2{i};
    data2 = can_train_ft2{i};
    data3 = mot_train_ft2{i};
    data4 = bat_train_ft2{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_train_data_ft2{i,1} = data1(1:100,:);
    cylind_train_targ_ft2{i,1} = data1(101:end,:);

    can_train_data_ft2{i,1} = data2(1:100,:);
    can_train_targ_ft2{i,1} = data2(101:end,:);

    mot_train_data_ft2{i,1} = data3(1:100,:);
    mot_train_targ_ft2{i,1} = data3(101:end,:);

    bat_train_data_ft2{i,1} = data4(1:100,:);
    bat_train_targ_ft2{i,1} = data4(101:end,:);
end

len_cyl_test = length(cylind_test_ft2);

% data1 = [];
% data2 = [];
% data3 = [];
% data4 = [];

for i = 1:len_cyl_test
    data1 = cylind_test_ft2{i};
    data2 = can_test_ft2{i};
    data3 = mot_test_ft2{i};
    data4 = bat_test_ft2{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_test_data_ft2{i,1} = data1(1:100,:);
    cylind_test_targ_ft2{i,1} = data1(101:end,:);

    can_test_data_ft2{i,1} = data2(1:100,:);
    can_test_targ_ft2{i,1} = data2(101:end,:);

    mot_test_data_ft2{i,1} = data3(1:100,:);
    mot_test_targ_ft2{i,1} = data3(101:end,:);

    bat_test_data_ft2{i,1} = data4(1:100,:);
    bat_test_targ_ft2{i,1} = data4(101:end,:);
end

%% Fault 3

[n_train,w_train] = size(norm_fault_train_3);
[n_test, w_test] = size(norm_fault_test_3);

cylind_train_data_ft3 = cell(1,1);
cylind_train_targ_ft3 = cell(1,1);
can_train_data_ft3 = cell(1,1);
can_train_targ_ft3 = cell(1,1);
mot_train_data_ft3 = cell(1,1);
mot_train_targ_ft3 = cell(1,1);
bat_train_data_ft3 = cell(1,1);
bat_train_targ_ft3 = cell(1,1);

cylind_test_data_ft3 = cell(1,1);
cylind_test_targ_ft3 = cell(1,1);
can_test_data_ft3 = cell(1,1);
can_test_targ_ft3 = cell(1,1);

mot_test_data_ft3 = cell(1,1);
mot_test_targ_ft3 = cell(1,1);
bat_test_data_ft3 = cell(1,1);
bat_test_targ_ft3 = cell(1,1);

data_temp = cell(4,1);

cylind_train_ft3 = norm_fault_train_3{1};
can_train_ft3 = norm_fault_train_3{2};
mot_train_ft3 = norm_fault_train_3{3};
bat_train_ft3 = norm_fault_train_3{4};

cylind_test_ft3 = norm_fault_test_3{1};
can_test_ft3 = norm_fault_test_3{2};
mot_test_ft3 = norm_fault_test_3{3};
bat_test_ft3 = norm_fault_test_3{4};

len_cyl = length(cylind_train_ft1);

for i = 1:len_cyl
    data1 = cylind_train_ft3{i};
    data2 = can_train_ft3{i};
    data3 = mot_train_ft3{i};
    data4 = bat_train_ft3{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_train_data_ft3{i,1} = data1(1:100,:);
    cylind_train_targ_ft3{i,1} = data1(101:end,:);

    can_train_data_ft3{i,1} = data2(1:100,:);
    can_train_targ_ft3{i,1} = data2(101:end,:);

    mot_train_data_ft3{i,1} = data3(1:100,:);
    mot_train_targ_ft3{i,1} = data3(101:end,:);

    bat_train_data_ft3{i,1} = data4(1:100,:);
    bat_train_targ_ft3{i,1} = data4(101:end,:);
end

len_cyl_test = length(cylind_test_ft3);


for i = 1:len_cyl_test
    data1 = cylind_test_ft3{i};
    data2 = can_test_ft3{i};
    data3 = mot_test_ft3{i};
    data4 = bat_test_ft3{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_test_data_ft3{i,1} = data1(1:100,:);
    cylind_test_targ_ft3{i,1} = data1(101:end,:);

    can_test_data_ft3{i,1} = data2(1:100,:);
    can_test_targ_ft3{i,1} = data2(101:end,:);

    mot_test_data_ft3{i,1} = data3(1:100,:);
    mot_test_targ_ft3{i,1} = data3(101:end,:);

    bat_test_data_ft3{i,1} = data4(1:100,:);
    bat_test_targ_ft3{i,1} = data4(101:end,:);
end

%% Fault 4

[n_train,w_train] = size(norm_fault_train_4);
[n_test, w_test] = size(norm_fault_test_4);

cylind_train_data_ft4 = cell(1,1);
cylind_train_targ_ft4 = cell(1,1);
can_train_data_ft4 = cell(1,1);
can_train_targ_ft4 = cell(1,1);
mot_train_data_ft4 = cell(1,1);
mot_train_targ_ft4 = cell(1,1);
bat_train_data_ft4 = cell(1,1);
bat_train_targ_ft4 = cell(1,1);

cylind_test_data_ft4 = cell(1,1);
cylind_test_targ_ft4 = cell(1,1);
can_test_data_ft4 = cell(1,1);
can_test_targ_ft4 = cell(1,1);

mot_test_data_ft4 = cell(1,1);
mot_test_targ_ft4 = cell(1,1);
bat_test_data_ft4 = cell(1,1);
bat_test_targ_ft4 = cell(1,1);

data_temp = cell(4,1);

cylind_train_ft4 = norm_fault_train_4{1};
can_train_ft4 = norm_fault_train_4{2};
mot_train_ft4 = norm_fault_train_4{3};
bat_train_ft4 = norm_fault_train_4{4};

cylind_test_ft4 = norm_fault_test_4{1};
can_test_ft4 = norm_fault_test_4{2};
mot_test_ft4 = norm_fault_test_4{3};
bat_test_ft4 = norm_fault_test_4{4};

len_cyl = length(cylind_train_ft4);

for i = 1:len_cyl
    data1 = cylind_train_ft4{i};
    data2 = can_train_ft4{i};
    data3 = mot_train_ft4{i};
    data4 = bat_train_ft4{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_train_data_ft4{i,1} = data1(1:100,:);
    cylind_train_targ_ft4{i,1} = data1(101:end,:);

    can_train_data_ft4{i,1} = data2(1:100,:);
    can_train_targ_ft4{i,1} = data2(101:end,:);

    mot_train_data_ft4{i,1} = data3(1:100,:);
    mot_train_targ_ft4{i,1} = data3(101:end,:);

    bat_train_data_ft4{i,1} = data4(1:100,:);
    bat_train_targ_ft4{i,1} = data4(101:end,:);
end

len_cyl_test = length(cylind_test_ft1);

for i = 1:len_cyl_test
    data1 = cylind_test_ft4{i};
    data2 = can_test_ft4{i};
    data3 = mot_test_ft4{i};
    data4 = bat_test_ft4{i};

    data_tmp = [data1 data2 data3 data4];

    cylind_test_data_ft4{i,1} = data1(1:100,:);
    cylind_test_targ_ft4{i,1} = data1(101:end,:);

    can_test_data_ft4{i,1} = data2(1:100,:);
    can_test_targ_ft4{i,1} = data2(101:end,:);

    mot_test_data_ft4{i,1} = data3(1:100,:);
    mot_test_targ_ft4{i,1} = data3(101:end,:);

    bat_test_data_ft4{i,1} = data4(1:100,:);
    bat_test_targ_ft4{i,1} = data4(101:end,:);
end


%% Normalize training & testing data
function norm_data = generate_normalized_signal(fault_train, mu_cell, sig_cell)
    for i = 1:4
        tot_data = fault_train{i};
        len_tot = length(tot_data);
        mu_tmp = mu_cell{i};
        sig_tmp = sig_cell{i};

        for j = 1:len_tot
            data_tmp = tot_data{j};
            data_tmp = (data_tmp - mu_tmp) ./ sig_tmp;

            tot_data{j} = data_tmp;
        end
        fault_train{i} = tot_data;
    end
    norm_data = fault_train;
end

%% main caller for fault signal generation 
function [fault_train, fault_lab_cell, fault_lab_array_cell] = ...
    generate_fault_signal(org_train,max_cell)

    fault_type_num = 8; % 1-Hardover, 2-Erractic, 3-Spike, 4-Drift    
    fault_train = org_train;
    fault_lab_cell = cell(4,1);
    fault_lab_array_cell = cell(4,1);
    for j = 1:4
        org_cell = org_train{j};
        len_cell = length(org_cell);
        fault_lab_cell_sub = cell(len_cell, 1);
        fault_lab_cell_sub_array = [];
        max_sub = max_cell{j};
        fault_cell = org_cell;

        for i = 1:len_cell
            org_data = org_cell{i};
            [len_s, dim] = size(org_data);
            % Choose how many fault channels
            fault_num_rand_seed = randi(dim);
            % Choose which fault channel
            fault_channel_rand_seed = sort(randperm(dim, fault_num_rand_seed));
            % Choose which fault type for the specific fault channel
            fault_type_rand_seed = randperm(fault_type_num, fault_num_rand_seed);

            fault_inject = inject_fault_func(org_data, fault_num_rand_seed,...
                fault_channel_rand_seed,fault_type_rand_seed, max_sub);

            fault_label = generate_fault_lable(dim, fault_channel_rand_seed, fault_type_rand_seed);

            fault_cell{i} = fault_inject;
            fault_lab_cell_sub{i} = fault_label;
            fault_lab_cell_sub_array(i,:) = fault_label;
        end
        fault_train{j} = fault_cell;
        fault_lab_cell{j} = fault_lab_cell_sub;
        fault_lab_array_cell{j} = fault_lab_cell_sub_array;
    end 

end


%%  Inject fault signal
function fault_array = inject_fault_func(org_data,fault_num_rand_seed,...
    fault_channel_rand_seed,fault_type_rand_seed, max_sub)

    fault_array = org_data;
    max_tmp = max_sub;
    for fau_num = 1:fault_num_rand_seed
        
        fault_channel_tmp = fault_channel_rand_seed(fau_num); % select channel
        fault_channel_data = fault_array(:, fault_channel_tmp);
%         size(fault_channel_data)

%         max_d = (max(fault_channel_data) + max_tmp(fault_channel_tmp))/2;
        max_d = max([max(fault_channel_data), max_tmp(fault_channel_tmp)/10]);
        
        fault_type_tmp = fault_type_rand_seed(fau_num);

        if fault_type_tmp == 1
            % perform hardover fault
            start_pt = randi([25,80]);
%             fault_channel_data(start_pt:end) = 1.1*max_tmp(fault_channel_tmp);
            fault_channel_data(start_pt:end) = 2*max_d;

        elseif fault_type_tmp == 2
            % perform erratic fault
            start_pt = randi([20,80]);
            range = start_pt:length(fault_channel_data);
            rand_n = rand(1,length(range))*2-1;
%             rand_n_fso = 0.2*max_tmp(fault_channel_tmp)*rand_n;
            rand_n_fso = 2*max_d*rand_n;
            fault_channel_data(start_pt:end) = fault_channel_data(start_pt:end) + rand_n_fso';

        elseif fault_type_tmp == 3
            % perform spike fault
            start_pt = randi([10,50]);
            rand_spike = sort(randperm(length(fault_channel_data),start_pt));
%             fault_channel_data(rand_spike) = fault_channel_data(rand_spike) + 0.2*max_tmp(fault_channel_tmp);
            fault_channel_data(rand_spike) = fault_channel_data(rand_spike) + 1.5*max_d;

        elseif fault_type_tmp == 4
            % perform drift fault
            start_pt = randi([25,50]);
%             fault_channel_data(end) = max_tmp(fault_channel_tmp);
%             slope = (max_tmp(fault_channel_tmp)-fault_channel_data(start_pt))/(length(fault_channel_data)-start_pt);
            fault_channel_data(end) = max_d;
            slope = (1.5*max_d-fault_channel_data(start_pt))/(length(fault_channel_data)-start_pt);

            for drift_k = 1:(length(fault_channel_data)-start_pt)
                fault_channel_data(start_pt+drift_k) = fault_channel_data(start_pt) + slope*drift_k;
            end

        elseif fault_type_tmp == 5
            fault_channel_data = awgn(fault_channel_data, 1, 'measured');

        elseif fault_type_tmp == 6
            fault_channel_data = 0;

        elseif fault_type_tmp == 7
%             size(fault_channel_data)
            fault_channel_data = fault_channel_data + sin(1:length(fault_channel_data))'/10;

        elseif fault_type_tmp == 8
            fault_channel_data = fault_channel_data + sawtooth(1:length(fault_channel_data))'/10;
        end
        fault_array(:,fault_channel_tmp) = fault_channel_data;
    end
end

%% Generate fault label
function fault_label = generate_fault_lable(dim, fault_channel_rand_seed, fault_type_rand_seed)
    fault_label = zeros(1,dim);
    fault_label(fault_channel_rand_seed) = fault_type_rand_seed;
end
