sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims.(sets{1}) = [256 256];
dims.(sets{2}) = [224 224];

contest.name = 'ILSVRC';


model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
model.paths.print_paths();
model.set_contest(contest);
 
% try
%     model.load_metadata();
% catch err
%     APP_LOG('warning',0,'%s',err.message);
%     model.build_metadata();
%     model.check_metadata();
%     model.save_metadata();
% end

% model.print_metadata();

model.objects.set_dims(dims);
try
	model.load_objects();
catch err
    APP_LOG('warning',0,'%s',err.message);
    model.build_objects();
    model.save_objects();
end

net = network();
net.set_model(model);
clear model;

%caffe---------------------------------------------------------
net.caffe.set_use_gpu(1);
net.caffe.set.device(0);
net.caffe.set_net_structure(fullfile(pwd,'caffe','prototxt','tst_dummy_quick.prototxt'));
net.caffe.init();

net.caffe.set_layer(1,gabor2D(7),0);            %layer_number / weights / bias value (uniform)
net.caffe.set_layer(2,random_bank_filters(5,5,32,256),0);
net.caffe.set_layer(3,random_bank_filters(3,3,256,384),0);
net.caffe.set_layer(4,random_bank_filters(3,3,384,384),0);
net.caffe.set_layer(5,random_bank_filters(3,3,384,256),0);

net.caffe.batch_factory.set_training_batch_method('random_segments',{4});
net.caffe.batch_factory.use_flipped(1);             %Use Horizontal flipped images during training
net.caffe.batch_factory.set_batch_size(128);        %How many images per batch.
net.caffe.batch_factory.set_input_size([224 224]);  %Input image size [HxW].
%caffe---------------------------------------------------------


net.set_batches_per_iter(2);                    %How many batches to perform 1 weight update 
net.set_validations_per_epoch(2);

net.train.set_constant_layers(1);
net.train.set_method('sgd');
net.train.method.init();
net.train.method.set_params({0.014142136,0.9,0.1,20*net.iters_per_epoch,0.0005});

net.validation.set_k(5);
net.set_max_iterations(100*net.iters_per_epoch); %STOP parameter - What is the maximum size of epochs to trigger a stop
net.set_target(0.25);                            %STOP parameter - What is the minimum validation error to trigger a stop
net.set_snapshot_time(240);                      %Time in minutes
net.start();