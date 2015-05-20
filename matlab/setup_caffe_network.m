%setup_network Sets up parameters of the network
 
%% Universal network parameters

    %Neural Network input size (squared)
    net.input_size.width       = 224;
    net.input_size.height      = 224;
    %flip objects? (0/1)
    net.flip                = 1;

%% Training parameters

    %Training options
        %momentum
        net.train.m             = 0.8;
        %learning rate
        net.train.lr            = 0.0135;
        %batch size for training
        net.train.batch_size    = 128;

    %Training stats
        %classification error
        net.train.error         = [];
        %training iteration
        net.train.iter          = 0;
    
%% Print INFO        

mode{1}='CPU';mode{2}='GPU';
APP_LOG('header',0,'NETWORK INFO');
APP_LOG('info',0,'Network input size [%d , %d]',net.input.size.width,net.input.size.height);
APP_LOG('info',0,'Use flipped objects: %d',net.flip);
APP_LOG('info',0,'Batch size: %d',net.input.batch.size);
%APP_LOG('info',0,'Training method: %s',net.train.method);
%APP_LOG('info',0,'Momentum: %f',net.train.m);
%APP_LOG('info',0,'Momentum: %f',net.train.lr);

APP_LOG('info',0,'Network info\n%25s: %d\n%25s: %d\n%25s: %f\n%25s: %d\n%25s: %d\n%25s: %d\n%25s: %d\n%25s: %s\n%25s: %s', ...
               'Network input size',net.NN_input_size,...
               'Use flipped images',net.flip,...
               'Momentum',net.train.m,...
               'Learning rate',net.train.lr,...
               'Batch size',net.train.batch_size,...
               'Training iterations',net.train.iter,...
               'Caffe number of layers',net.caffe.nlayers,...
               'Caffe model prototxt',net.caffe.prototxt,...
               'Caffe mode',char(mode{net.caffe.use_gpu+1}));
           
%% Garbage collection
clear i;
clear mode;
net.caffe.reset();