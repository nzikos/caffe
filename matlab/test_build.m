sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims  = [224 224];
contest.name = 'ILSVRC';


model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
%model.paths.print_paths();
model.set_contest(contest);

try
    model.load_metadata();
catch err
    APP_LOG('warning',0,'%s',err.message);
    model.build_metadata();
    model.check_metadata();
    model.save_metadata();
end

%model.print_metadata();

model.set_objects(dims);

try
	model.load_objects();
catch err
    APP_LOG('warning',0,'%s',err.message);
    model.build_objects();
    model.save_objects();
end

net = network(model);

net.caffe.set_use_gpu(1);
net.caffe.set_net_structure(fullfile(pwd,'caffe','prototxt','tst_dummy_quick.prototxt'));
net.caffe.set_batch_size(64);  %How many images per batch
net.caffe.init();

net.caffe.set_layer(1,gabor2D(15),0); %layer_number / weights / bias value (uniform)

net.set_batches_per_iter(2);        %How many batches to perform 1 weight update 
net.set_validations_per_epoch(4);
net.set_max_epochs(200);            %STOP parameter - What is the maximum size of epochs to trigger a stop
net.set_target(0.25);               %STOP parameter - What is the minimum validation error to trigger a stop

net.train.set_method('sgd',{0.000035,0.8});

net.start();