net = network();
net.load_snapshot(fullfile('/','media','alexis','SSD','CACHE','snapshots','tic_toc_snapshot.mat'));
net.caffe.init();
net.start();