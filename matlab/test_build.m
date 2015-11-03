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

%% SET NETWORK STRUCTURE
net_struct = NET_STRUCTURE(cache);
net_struct.set_batch_size('train',128);
net_struct.set_batch_size('validation',128);
net_struct.set_batch_size('test',10);
net_struct.set_input_object_dims(227,227,3);
net_struct.set_labels_length(1);
net_struct.add_CONV_layer   ('conv1',96,[11 11],[4 4],0,true,'gaussian',0,0.01,1,'constant',1,1,1);
net_struct.add_ACTIV_layer  ('relu1','relu');
%net_struct.add_LRN_layer    ('lrn1' ,3,0.00005,0.75,'within_channel');
net_struct.add_POOL_layer   ('pool1','MAX',[3 3],2,[0 0]);
net_struct.add_MVN_layer    ('mvn1','true','true');
net_struct.add_CONV_layer   ('conv2',256,5,1,2,true,'gaussian',0,0.01,1,'constant',0,0,1);
net_struct.add_ACTIV_layer  ('relu2','relu');
%net_struct.add_LRN_layer    ('lrn2' ,3,0.00005,0.75,'within_channel');
net_struct.add_POOL_layer   ('pool2','MAX',3,2,0);
net_struct.add_MVN_layer    ('mvn2','true','true');
net_struct.add_CONV_layer   ('conv3',384,3,1,1,true,'gaussian',0,0.01,1,'constant',0,0,1);
net_struct.add_ACTIV_layer  ('relu3','relu');
net_struct.add_MVN_layer    ('mvn3','true','true');
net_struct.add_CONV_layer   ('conv4',384,3,1,1,true,'gaussian',0,0.01,1,'constant',0,0,1);
net_struct.add_ACTIV_layer  ('relu4','relu');
net_struct.add_MVN_layer    ('mvn4','true','true');
net_struct.add_CONV_layer   ('conv5',256,3,1,1,true,'gaussian',0,0.01,1,'constant',0,0,1);
net_struct.add_ACTIV_layer  ('relu5','relu');
net_struct.add_POOL_layer   ('pool5','MAX',3,2,0);
net_struct.add_IP_layer     ('fc6'  ,4096,true,'gaussian',0,0.01,1,'constant',0,0,1);
net_struct.add_ACTIV_layer  ('relu6','relu');
net_struct.add_DROPOUT_layer('drop6',0.7);
net_struct.add_IP_layer     ('fc7'  ,4096,true,'gaussian',0,0.01,1,'constant',0,0,1);
net_struct.add_ACTIV_layer  ('relu7','relu');
net_struct.add_DROPOUT_layer('drop7',0.7);
net_struct.add_IP_layer     ('fc8'  ,200,true,'gaussian',0,0.01,1,'constant',0,0,1); 
net_struct.add_OUTPUT_ERROR_layer('SOFTMAX_LOSS');
%% -------------------------------------------------

net = network();
net.set_model(model);
clear model;
%% SET CAFFE
net.caffe.set_use_gpu(1,0);                                % (enable/disable, device_id) device_id is zero-based
net.caffe.set_structure(net_struct);
net.caffe.init('train');

%                   layer_number/attribute/weights/bias value (uniform)
%net.caffe.set_layer(1,gabor2D(11,3),0);
%net.caffe.set_layer(2,random_bank_filters(5,5,32,256),0);
%net.caffe.set_layer(3,random_bank_filters(3,3,256,384),0);
%net.caffe.set_layer(4,random_bank_filters(3,3,384,384),0);
%net.caffe.set_layer(5,random_bank_filters(3,3,384,256),0);

net.batch_factory.set_async_queue_size(5);
net.batch_factory.random_segmentation(1);            %frequency per batch [0-1]
net.batch_factory.rotation([-20 20],0);              %Rotate 0% of batch with a random angle between [-20,20].
net.batch_factory.projections(0);                    %Projections frequency(hardcoded params, TESTS PENDING)
net.batch_factory.flipped(0.5);                      %Use horizontal flipped images during training
net.batch_factory.normalize_input('subtract_means_normalize_variances');  %[X - E(D)]/std(D)
%net.batch_factory.normalize_input('zero_one_scale');
%caffe---------------------------------------------------------

net.set_batches_per_iter(1);                  %How many batches to perform 1 weight update 

%net.set_validations_per_epoch(40);
net.set_validation_interval(3800);

net.train.set_method('user_defined',{net.validation,net.exit_train});
net.train.method.set_learning_params({0.01,0.9,0.2,0.0005,2,1});

%net.train.set_method('SGD');
%net.train.method.set_learning_params({0.01,0.9,0.1,100000,0.0005});

net.validation.set_k(5);
net.validation.set_best_target('Average','top1')
%net.set_max_iterations(350000); %STOP parameter - What is the maximum size of iterations to trigger a stop
net.set_max_iterations(inf);                     %STOP parameter - What is the maximum size of epochs to trigger a stop
net.set_snapshot_time(8*60);                     %Save a snapshot of the net every (time in minutes)
net.set_display(200);                            %Display training stats every x iterations
net.fetch_train_error(0);                  %Enable(1)/Disable(0) computation of train error to increase speed in case of latency due to caffe
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