%% Light Train
falut4_ratio = 0.1;        %controls how many fault signals to the testing set

ind_train = randperm(length(cylind_train_targ_ft4));
ind_train = ind_train(1:ceil(length(ind_train)*falut4_ratio));

ind_test = randperm(length(cylind_test_targ_ft4));
ind_test = ind_test(1:ceil(length(ind_test)*falut4_ratio));

clnd_train_ft_dt = cylind_train_data_ft4(ind_train);
clnd_train_ft_tg = cylind_train_targ(ind_train);
clnd_train_dt_comb = [cylind_train_data; clnd_train_ft_dt];
clnd_train_tg_comb = [cylind_train_targ; clnd_train_ft_tg];
clnd_train_ft_lb = fault_lab_train_array_4{1};
clnd_train_ft_lb = clnd_train_ft_lb(ind_train,:);
clnd_train_lb_comb = [zeros(length(cylind_train_targ_ft4),4); clnd_train_ft_lb];

%% Light Test
clnd_test_ft_dt = cylind_test_data_ft4;
clnd_test_ft_tg = cylind_test_targ;
clnd_test_dt_comb = [cylind_test_data; clnd_test_ft_dt];
clnd_test_tg_comb = [cylind_test_targ; clnd_test_ft_tg];
clnd_test_ft_lb = fault_lab_test_array_4{1};
clnd_test_lb_comb = [zeros(length(cylind_test_targ_ft4),4); clnd_test_ft_lb];


%% Dark Train

dark_clind_train_lb = [fault_lab_train_array_1{1};fault_lab_train_array_2{1}; fault_lab_train_array_3{1}];

dark_clind_train_dt = [cylind_train_data_ft1; cylind_train_data_ft2; cylind_train_data_ft3];

dark_clind_train_tg = [cylind_train_targ; cylind_train_targ; cylind_train_targ];

dark_clind_train_recon_tg = [cylind_train_data; cylind_train_data; cylind_train_data];

%% Dark Test

dark_clind_test_lb = [fault_lab_test_array_1{1};fault_lab_test_array_2{1}; fault_lab_test_array_3{1}];

dark_clind_test_dt = [cylind_test_data_ft1; cylind_test_data_ft2; cylind_test_data_ft3];

dark_clind_test_tg = [cylind_test_targ; cylind_test_targ; cylind_test_targ];

dark_clind_test_recon_tg = [cylind_test_data; cylind_test_data; cylind_test_data];


%% Save data

if 1
    save light-dark-clinder.mat clnd_train_dt_comb clnd_train_tg_comb clnd_train_lb_comb  ...
        clnd_test_dt_comb clnd_test_tg_comb clnd_test_lb_comb  ...
        dark_clind_test_lb dark_clind_test_dt dark_clind_test_tg   dark_clind_test_recon_tg ... 
        dark_clind_train_lb dark_clind_train_dt dark_clind_train_tg dark_clind_train_recon_tg   

end
