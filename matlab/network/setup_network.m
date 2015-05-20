function net = setup_network()
%% Setup network
    APP_LOG('header',0,'Setup network');
%% Input Parameters
    net.input.size.width    = 224;
    net.input.size.height   = 224;    

%% caffe
    net.caffe = setup_caffe();
    net.caffe = init_caffe(net.caffe);
    
end

