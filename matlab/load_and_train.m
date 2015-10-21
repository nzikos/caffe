net = network();
net.load_snapshot(fullfile('/','media','alexis','SSD','CACHE','snapshots','snapshot_20_10_2015_16_28.mat'));
%net.set_display(1);                            %Display training stats every x iterations
%net.set_snapshot_time(120);                       %Save a snapshot of the net every (time in minutes)
net.start();

%Testing matlab git commit