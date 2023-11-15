%% Light Train
falut4_ratio = 0.1;

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

can_train_ft_dt = can_train_data_ft4(ind_train);
can_train_ft_tg = can_train_targ(ind_train);
can_train_dt_comb = [can_train_data; can_train_ft_dt];
can_train_tg_comb = [can_train_targ; can_train_ft_tg];
can_train_ft_lb = fault_lab_train_array_4{2};
can_train_ft_lb = can_train_ft_lb(ind_train,:);
can_train_lb_comb = [zeros(length(cylind_train_targ_ft4),4); can_train_ft_lb];

mot_train_ft_dt = mot_train_data_ft4(ind_train);
mot_train_ft_tg = mot_train_targ(ind_train);
mot_train_dt_comb = [mot_train_data; mot_train_ft_dt];
mot_train_tg_comb = [mot_train_targ; mot_train_ft_tg];
mot_train_ft_lb = fault_lab_train_array_4{3};
mot_train_ft_lb = mot_train_ft_lb(ind_train,:);
mot_train_lb_comb = [zeros(length(cylind_train_targ_ft4),3); mot_train_ft_lb];

bat_train_ft_dt = bat_train_data_ft4(ind_train);
bat_train_ft_tg = bat_train_targ(ind_train);
bat_train_dt_comb = [bat_train_data; bat_train_ft_dt];
bat_train_tg_comb = [bat_train_targ; bat_train_ft_tg];
bat_train_ft_lb = fault_lab_train_array_4{4};
bat_train_ft_lb = bat_train_ft_lb(ind_train,:);
bat_train_lb_comb = [zeros(length(cylind_train_targ_ft4),3); bat_train_ft_lb];
%% Light Test
clnd_test_ft_dt = cylind_test_data_ft4;
clnd_test_ft_tg = cylind_test_targ;
clnd_test_dt_comb = [cylind_test_data; clnd_test_ft_dt];
clnd_test_tg_comb = [cylind_test_targ; clnd_test_ft_tg];
clnd_test_ft_lb = fault_lab_test_array_4{1};
clnd_test_lb_comb = [zeros(length(cylind_test_targ_ft4),4); clnd_test_ft_lb];

can_test_ft_dt = can_test_data_ft4;
can_test_ft_tg = can_test_targ;
can_test_dt_comb = [can_test_data; can_test_ft_dt];
can_test_tg_comb = [can_test_targ; can_test_ft_tg];
can_test_ft_lb = fault_lab_test_array_4{2};
can_test_lb_comb = [zeros(length(can_test_targ_ft4),4); can_test_ft_lb];

mot_test_ft_dt = mot_test_data_ft4;
mot_test_ft_tg = mot_test_targ;
mot_test_dt_comb = [mot_test_data; mot_test_ft_dt];
mot_test_tg_comb = [mot_test_targ; mot_test_ft_tg];
mot_test_ft_lb = fault_lab_test_array_4{3};
mot_test_lb_comb = [zeros(length(mot_test_targ_ft4),3); mot_test_ft_lb];

bat_test_ft_dt = bat_test_data_ft4;
bat_test_ft_tg = bat_test_targ;
bat_test_dt_comb = [bat_test_data; bat_test_ft_dt];
bat_test_tg_comb = [bat_test_targ; bat_test_ft_tg];
bat_test_ft_lb = fault_lab_test_array_4{4};
bat_test_lb_comb = [zeros(length(bat_test_targ_ft4),3); bat_test_ft_lb];


%% Dark Train

dark_clind_train_lb = [fault_lab_train_array_1{1};fault_lab_train_array_2{1}; fault_lab_train_array_3{1}];
dark_can_train_lb = [fault_lab_train_array_1{2};fault_lab_train_array_2{2}; fault_lab_train_array_3{2}];
dark_mot_train_lb = [fault_lab_train_array_1{3};fault_lab_train_array_2{3}; fault_lab_train_array_3{3}];
dark_bat_train_lb = [fault_lab_train_array_1{4};fault_lab_train_array_2{4}; fault_lab_train_array_3{4}];

dark_clind_train_dt = [cylind_train_data_ft1; cylind_train_data_ft2; cylind_train_data_ft3];
dark_can_train_dt = [can_train_data_ft1; can_train_data_ft2; can_train_data_ft3];
dark_mot_train_dt = [mot_train_data_ft1; mot_train_data_ft2; mot_train_data_ft3];
dark_bat_train_dt = [bat_train_data_ft1; bat_train_data_ft2; bat_train_data_ft3];

dark_clind_train_tg = [cylind_train_targ; cylind_train_targ; cylind_train_targ];
dark_can_train_tg = [can_train_targ; can_train_targ; can_train_targ];
dark_mot_train_tg = [mot_train_targ; mot_train_targ; mot_train_targ];
dark_bat_train_tg = [bat_train_targ; bat_train_targ; bat_train_targ];

dark_clind_train_recon_tg = [cylind_train_data; cylind_train_data; cylind_train_data];
dark_can_train_recon_tg = [can_train_data; can_train_data; can_train_data];
dark_mot_train_recon_tg = [mot_train_data; mot_train_data; mot_train_data];
dark_bat_train_recon_tg = [bat_train_data; bat_train_data; bat_train_data];

%% Dark Test

dark_clind_test_lb = [fault_lab_test_array_1{1};fault_lab_test_array_2{1}; fault_lab_test_array_3{1}];
dark_can_test_lb = [fault_lab_test_array_1{2};fault_lab_test_array_2{2}; fault_lab_test_array_3{2}];
dark_mot_test_lb = [fault_lab_test_array_1{3};fault_lab_test_array_2{3}; fault_lab_test_array_3{3}];
dark_bat_test_lb = [fault_lab_test_array_1{4};fault_lab_test_array_2{4}; fault_lab_test_array_3{4}];

dark_clind_test_dt = [cylind_test_data_ft1; cylind_test_data_ft2; cylind_test_data_ft3];
dark_can_test_dt = [can_test_targ_ft1; can_test_targ_ft2; can_test_targ_ft3];
dark_mot_test_dt = [mot_test_data_ft1; mot_test_data_ft2; mot_test_data_ft3];
dark_bat_test_dt = [bat_test_data_ft1; bat_test_data_ft2; bat_test_data_ft2];

dark_clind_test_tg = [cylind_test_targ; cylind_test_targ; cylind_test_targ];
dark_can_test_tg = [can_test_targ; can_test_targ; can_test_targ];
dark_mot_test_tg = [mot_test_targ; mot_test_targ; mot_test_targ];
dark_bat_test_tg = [bat_test_targ; bat_test_targ; bat_test_targ];

dark_clind_test_recon_tg = [cylind_test_data; cylind_test_data; cylind_test_data];
dark_can_test_recon_tg = [can_test_data; can_test_data; can_test_data];
dark_mot_test_recon_tg = [mot_test_data; mot_test_data; mot_test_data];
dark_bat_test_recon_tg = [bat_test_data; bat_test_data; bat_test_data];


%% Save data

if 1
    save light-dark.mat clnd_train_dt_comb clnd_train_tg_comb clnd_train_lb_comb can_train_dt_comb can_train_tg_comb can_train_lb_comb ...
        mot_train_dt_comb mot_train_tg_comb mot_train_lb_comb bat_train_dt_comb bat_train_tg_comb bat_train_lb_comb ...
        clnd_test_dt_comb clnd_test_tg_comb clnd_test_lb_comb can_test_dt_comb can_test_tg_comb can_test_lb_comb ...
        mot_test_dt_comb mot_test_tg_comb mot_test_lb_comb bat_test_dt_comb bat_test_tg_comb bat_test_lb_comb ...
        dark_clind_test_lb dark_can_test_lb dark_mot_test_lb dark_bat_test_lb dark_clind_test_dt dark_can_test_dt dark_mot_test_dt dark_bat_test_dt...
        dark_clind_test_tg dark_can_test_tg dark_mot_test_tg dark_bat_test_tg dark_clind_test_recon_tg dark_can_test_recon_tg dark_mot_test_recon_tg dark_bat_test_recon_tg ... 
        dark_clind_train_lb dark_can_train_lb dark_mot_train_lb dark_bat_train_lb dark_clind_train_dt dark_can_train_dt dark_mot_train_dt dark_bat_train_dt ...
        dark_clind_train_tg dark_can_train_tg dark_mot_train_tg dark_bat_train_tg dark_clind_train_recon_tg dark_can_train_recon_tg dark_mot_train_recon_tg dark_bat_train_recon_tg
    
    
    save light-dark-clinder.mat clnd_train_dt_comb clnd_train_tg_comb clnd_train_lb_comb  ...
        clnd_test_dt_comb clnd_test_tg_comb clnd_test_lb_comb  ...
        dark_clind_test_lb dark_clind_test_dt dark_clind_test_tg   dark_clind_test_recon_tg ... 
        dark_clind_train_lb dark_clind_train_dt dark_clind_train_tg dark_clind_train_recon_tg   

end