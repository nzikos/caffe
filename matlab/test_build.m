sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims.(sets{1}) = [256 256];
dims.(sets{2}) = [227 227];
dataset.name = 'ILSVRC_DET';

%% SET MODEL
model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
model.paths.print_paths();
model.set_dataset(dataset);
model.objects.set_dims(dims);
model.objects.compute_mean_std(1);

model.load_objects();

% s = RandStream('mt19937ar','Seed','shuffle');
% RandStream.setGlobalStream(s);

%% SET NETWORK STRUCTURE
net_struct = NET_STRUCTURE(cache);
net_struct.set_batch_size('train',128);
net_struct.set_batch_size('validation',128);
net_struct.set_batch_size('test',128);
net_struct.set_objects_dims(227,227,3);

%LAYER-1
net_struct.add_layer('Convolution' ,{96,11,4,0,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-2
net_struct.add_layer('MVN'         ,{true,true});
net_struct.add_layer('Convolution' ,{256,5,1,2,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-3
net_struct.add_layer('MVN'         ,{true,true});
net_struct.add_layer('Convolution' ,{384,3,1,1,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
%LAYER-4
net_struct.add_layer('MVN'         ,{true,true});
net_struct.add_layer('Convolution' ,{384,3,1,1,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
%LAYER-5
%net_struct.add_layer('MVN'         ,{true,true});
net_struct.add_layer('Convolution' ,{256,3,1,1,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-6
net_struct.add_layer('InnerProduct',{4096,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('Dropout'     ,{0.7});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
%LAYER-7
net_struct.add_layer('InnerProduct',{4096,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('Dropout'     ,{0.7});
net_struct.add_layer('PReLU'       ,{false},{[0.5,0.1, 0]},{'constant',0.25});
%LAYER-8
net_struct.add_layer('InnerProduct',{200,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('Output'      ,{'SoftmaxWithLoss'});
%% -------------------------------------------------

net = network(model,net_struct);
clear model;
%% SET CAFFE
net.caffe.set_use_gpu(0); %device_id (zero-based) if device_id < 0 then CPU execution
net.caffe.init('train');

net.batch_factory.set_async_queue_size(5);
net.batch_factory.crop(0.4);                  %frequency per batch [0-1]
net.batch_factory.rotate([-20 20],0.3);       %Rotate 0% of batch with a random angle between [-20,20].
net.batch_factory.skew([-0.35 0.35],0.3);
net.batch_factory.projections(0);             %Projections frequency(hardcoded params, TESTS PENDING)
net.batch_factory.use_flipped_samples(1);     %Use horizontal flipped images during training
%net.batch_factory.normalize_input('subtract_means');  %[X - E(D)]/std(D)
net.batch_factory.normalize_input('subtract_means_normalize_variances');  %[X - E(D)]/std(D)
%net.batch_factory.normalize_input('zero_one_scale');
%caffe---------------------------------------------------------

net.set_batches_per_iter(1);                  %How many batches to perform 1 weight update 

%net.set_validations_per_epoch(40);
net.set_validation_interval(1500);

net.train.set_method('user_defined',{net.validation,net.exit_train});
net.train.method.set_learning_params({0.015,0.9,0.3,0.0005,2,1});

%net.train.set_method('SGD');
%net.train.method.set_learning_params({0.01,0.9,0.1,100000,0.0005});

net.validation.set_k(5);
net.validation.set_best_target('Average','top1')
%net.set_max_iterations(350000); %STOP parameter - What is the maximum size of iterations to trigger a stop
net.set_max_iterations(inf);                     %STOP parameter - What is the maximum size of epochs to trigger a stop
net.set_snapshot_time(8*60);                     %Save a snapshot of the net every (time in minutes)
net.set_display(200);                            %Display training stats every x iterations
net.fetch_train_error(1);                  %Enable(1)/Disable(0) computation of train error to increase speed in case of latency due to caffe
net.start();

%% DEPLOY
clear net;

deploy_net = deployed_model2(fullfile(cache,'best_model.mat'));
deploy_net.caffe.set_use_gpu(1,0);
deploy_net.caffe.init('test');
deploy_net.caffe.reset_object_input('test',128);
deploy_net.caffe.set_phase('test');

%deploy_net.load_test_imdb(fullfile(imdb,'test'),'.JPEG');
deploy_net.load_validation_metadata(fullfile(cache,'meta.mat'));

deploy_net.set_val_bboxes_dir(fullfile(meta,sets{2}));
deploy_net.set_overlap(0.5); %values close to 1 lead to heavy overlap
deploy_net.set_segments_bounds([1 5]);
deploy_net.set_activation_threshold(0.7);

deploy_net.compute('validation');
%net.compute('test');