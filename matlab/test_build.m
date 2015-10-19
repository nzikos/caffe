sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims.(sets{1}) = [256 256];
dims.(sets{2}) = [227 227];

contest.name = 'ILSVRC';


model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
model.paths.print_paths();
model.set_contest(contest);

try
    model.load_metadata();
catch err
    APP_LOG('warning','%s',err.message);
    model.build_metadata();
    model.check_metadata();
    model.save_metadata();
end

% model.print_metadata();

model.objects.set_dims(dims);
model.objects.compute_mean_std(1);
try
	model.load_objects();
catch err
    APP_LOG('warning','%s',err.message);
    model.build_objects();
    model.save_objects();
end

net = network();
net.set_model(model);
clear model;

%caffe---------------------------------------------------------
net.caffe.set_use_gpu(1,0);                                % (enable/disable, device_id)
net.caffe.structure.set_batch_size('train',128);
net.caffe.structure.set_batch_size('validation',128);
net.caffe.structure.set_batch_size('test',10);
net.caffe.structure.set_input_object_dims(3,227,227);
net.caffe.structure.set_labels_length(1);

net.caffe.structure.add_CONV_layer   ('conv1',32,11,0,4,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu1','relu');
%net.caffe.structure.add_LRN_layer    ('lrn1' ,3,0.00005,0.75,'within_channel');
net.caffe.structure.add_POOL_layer   ('pool1','MAX',3,0,2);
net.caffe.structure.add_CONV_layer   ('conv2',256,5,2,1,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu2','relu');
%net.caffe.structure.add_LRN_layer    ('lrn2' ,3,0.00005,0.75,'within_channel');
net.caffe.structure.add_POOL_layer   ('pool2','MAX',3,0,2);
net.caffe.structure.add_CONV_layer   ('conv3',384,3,1,1,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu3','relu');
net.caffe.structure.add_CONV_layer   ('conv4',384,3,1,1,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu4','relu');
net.caffe.structure.add_CONV_layer   ('conv5',256,3,1,1,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu5','relu');
net.caffe.structure.add_POOL_layer   ('pool5','MAX',3,0,2);
net.caffe.structure.add_IP_layer     ('fc6'  ,4096,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu6','relu');
net.caffe.structure.add_DROPOUT_layer('drop6',0.5);
net.caffe.structure.add_IP_layer     ('fc7'  ,4096,true,'gaussian',0.01,1,'constant',0,1);
net.caffe.structure.add_ACTIV_layer  ('relu7','relu');
net.caffe.structure.add_DROPOUT_layer('drop7',0.5);
net.caffe.structure.add_IP_layer     ('fc8'  ,200,true,'gaussian',0.01,1,'constant',0,1); 

net.caffe.structure.add_OUTPUT_ERROR_layer('SOFTMAX_LOSS');
net.caffe.init('train');

net.caffe.set_layer(1,gabor2D(11,3),0);            %layer_number / weights / bias value (uniform)
%net.caffe.set_layer(2,random_bank_filters(5,5,32,256),0);
%net.caffe.set_layer(3,random_bank_filters(3,3,256,384),0);
% net.caffe.set_layer(4,random_bank_filters(3,3,384,384),0);
% net.caffe.set_layer(5,random_bank_filters(3,3,384,256),0);

net.caffe.batch_factory.use_random_segmentation(1,1);        %1-enable/0-disable | 2nd parameter frequency per batch [0-1]
net.caffe.batch_factory.use_rotation(0,[-20 20],0.4);        %Use rotations. Rotate 40% of batch with a random angle between [-10,10].
net.caffe.batch_factory.use_projections(0,0);                %Use projections (hardcoded params, TESTS PENDING)
net.caffe.batch_factory.use_flipped(1);                      %Use horizontal flipped images during training
net.caffe.batch_factory.normalize_batches(0);                %[X - E(D)]/std(D)
%caffe---------------------------------------------------------


net.set_batches_per_iter(1);                  %How many batches to perform 1 weight update 
net.set_validations_per_epoch(1);

net.train.set_constant_layers(1);             %[1 2 3 4 5] for multiple constant layers

net.train.set_method('user_defined',{net.validation,net.exit});
net.train.method.set_params({0.01,0.9,0.1,0.0005,4,1});

%net.train.set_method('SGD');
%net.train.method.set_params({0.01,0.9,0.1,100000,0.0005});

net.validation.set_k(5);
net.validation.set_best_target('Average','error')
%net.set_max_iterations(350000); %STOP parameter - What is the maximum size of iterations to trigger a stop
net.set_max_iterations(inf);                     %STOP parameter - What is the maximum size of epochs to trigger a stop
net.set_snapshot_time(8*60);                     %Save a snapshot of the net every (time in minutes)
net.set_display(400);                            %Display training stats every x iterations
net.set_compute_train_error(0);                  %Enable(1)/Disable(0) computation of train error to increase speed in case of latency due to caffe
net.start();