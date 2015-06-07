sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims  = [224 224];
contest.name = 'ILSVRC';


model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
% %model.paths.print_paths();
model.set_contest(contest);
% 
% try
%     model.load_metadata();
% catch err
%     APP_LOG('warning',0,'%s',err.message);
%     model.build_metadata();
%     model.check_metadata();
%     model.save_metadata();
% end
% 
% %model.print_metadata();
% 
model.set_objects(dims);

try
	model.load_objects();
catch err
    APP_LOG('warning',0,'%s',err.message);
    model.build_objects();
    model.save_objects();
end

net = network(model);
clear model;

net.caffe.set_use_gpu(1);
net.caffe.set_net_structure(fullfile(pwd,'caffe','prototxt','tst_dummy_quick.prototxt'));
net.caffe.set_batch_size(256);  %How many images per batch
net.caffe.init();

net.caffe.set_layer(1,gabor2D(7),0); %layer_number / weights / bias value (uniform)
% net.caffe.set_layer(2,random_bank_filters(5,5,32,256),0);
% net.caffe.set_layer(3,random_bank_filters(3,3,256,384),0);
% net.caffe.set_layer(4,random_bank_filters(3,3,384,384),0);
% net.caffe.set_layer(5,random_bank_filters(3,3,384,256),0);

net.set_batches_per_iter(4);        %How many batches to perform 1 weight update 
net.set_validations_per_epoch(2);
net.set_max_epochs(100);            %STOP parameter - What is the maximum size of epochs to trigger a stop
net.set_target(0.25);               %STOP parameter - What is the minimum validation error to trigger a stop

net.train.set_method('sgd',{0.0282842,0.9,0.1,30*net.iters_per_epoch,0.0005});
net.train.set_constant_layers(1);
net.start();