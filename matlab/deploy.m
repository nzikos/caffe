%% DEPRECATED 
% meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
% imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
% cache = fullfile('/','media','alexis','SSD','CACHE');
% 
% 
% net = deployed_model2(fullfile(cache,'best_model.mat'));
% net.caffe.set_use_gpu(1,0);
% net.caffe.init('test');
% net.caffe.reset_object_input('test',128);
% net.caffe.set_phase('test');
% 
% %net.load_test_imdb(fullfile(imdb,'test'),'.JPEG');
% net.load_validation_metadata(fullfile(cache,'meta.mat'));
% 
% net.set_val_bboxes_dir(fullfile('/','media','alexis','SSD','XML','validation'));
% net.set_overlap(0.5); %values close to 1 lead to heavy overlap
% net.set_segments_bounds([1 5]);
% net.set_activation_threshold(0.7);
% 
% net.compute('validation');
% %net.compute('test');
