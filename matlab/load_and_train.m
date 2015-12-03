model = extraction_model();
net = network(model,fullfile('/media/alexis/SSD/CACHE/snapshots/snapshot_24_11_2015_10_18.mat'));
net.start();