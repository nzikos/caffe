net = network();
net.load_snapshot(fullfile('/media/alexis/SSD/CACHE/snapshots/snapshot_02_11_2015_01_20.mat'));
%net.set_display(1);                            %Display training stats every x iterations
%net.set_snapshot_time(120);                       %Save a snapshot of the net every (time in minutes)
net.start();

%Testing matlab git commit