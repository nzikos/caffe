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
model.set_dims(dims);
model.compute_mean_std(1);

model.load_objects();
% s = RandStream('mt19937ar','Seed','shuffle');
% RandStream.setGlobalStream(s);

%% SET NETWORK STRUCTURE
net_struct = NET_STRUCTURE(cache);
net_struct.use_in_place(1);
net_struct.set_batch_size('train',128);
net_struct.set_batch_size('validation',128);
net_struct.set_batch_size('test',128);
net_struct.set_objects_dims(227,227,3);

%LAYER-1
net_struct.add_layer('Convolution' ,{96,11,4,0,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-2
%net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{256,5,1,2,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-3
net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{384,3,1,1,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
%LAYER-4
net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{384,3,1,1,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
%LAYER-5
%net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('Convolution' ,{256,3,1,1,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
net_struct.add_layer('Pooling'     ,{'MAX',3,2,0});
%LAYER-6
%net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('InnerProduct',{4096,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('Dropout'     ,{0.5});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
%LAYER-7
%net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('InnerProduct',{4096,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('Dropout'     ,{0.5});
net_struct.add_layer('PReLU'       ,{false},{[0, 0.1, 0]},{'constant',0.25});
%net_struct.add_layer('Activation'  ,{'ReLU'});
%LAYER-8
%net_struct.add_layer('MVN'         ,{false,true});
net_struct.add_layer('InnerProduct',{200,true},{[1,1,1],[1,1,1]},{{'gaussian',[0 0.01]},{'constant',0}});
net_struct.add_layer('Output'      ,{'SoftmaxWithLoss'});
%% -------------------------------------------------

net = network(model,net_struct);
%clear model net_struct;
%% SET CAFFE
net.caffe.set_use_gpu(0); %device_id (zero-based) if device_id < 0 then CPU execution

%% SET BATCH FACTORY
net.batch_factory.set_sampling_method(true,ones(1,200)/200);
%net.batch_factory.set_sampling_method(false,model.get_class_frequencies('train'));
net.batch_factory.set_async_queue_size(5);
net.batch_factory.set_tform('crop',{1});
net.batch_factory.set_tform('rotate',{[-20, 20], 0});
net.batch_factory.set_tform('skew',{[-0.35, 0.35], 0.0});
net.batch_factory.set_tform('projections',{0});%Projections frequency(hardcoded params, TESTS PENDING)
net.batch_factory.set_tform('flip',{true});

net.batch_factory.set_norm_type('subtract_means_normalize_variances'); %[X - E(D)]/std(D)
%net.batch_factory.set_norm_type('subtract_means');                     %[X - E(D)]
%net.batch_factory.set_norm_type('zero_one_scale');

%% SET TRAIN/VALIDATION/MONITOR
net.train.set('SGD',{0.9,0.01,0.0005});
%net.train.set('ADAGRAD',{0.1,eps});
net.train.set_batches_per_iter(1);         %How many batches to perform 1 weight update 
net.train.fetch_error(1);  	               %Enable(1)/Disable(0) computation of train error to increase speed in case of latency due to caffe

%net.lr_policy.set('step',{0.1,300});
net.lr_policy.set('heuristic',{0.1,2,1});

net.validation.set_k(5);
net.validation.set_best_target('Average','top1')

net.set_validation_interval(3500);
%net.set_max_iterations(350000); %STOP parameter - What is the maximum size of iterations to trigger a stop
%DEFAULT: net.set_max_iterations(inf);     %STOP parameter - What is the maximum size of epochs to trigger a stop

net.set_snapshot_time(8*60);     %Save a snapshot of the net every (time in minutes)
net.set_display(200);              %Display training stats every x iterations

net.monitor(1,'parameters',{'weights','bias'},{'mean_std'});
net.monitor(0,'gradients',{'weights','bias'},{'mean_std'});

net.start();