function caffe = setup_caffe()
%% Caffe related parameters and functions

    caffe.use_gpu            = 1;
    caffe.prototxt           = 'caffe/prototxt/dummy_quick.prototxt';
    
    %% SETTERS
    caffe.set.phase.train    = @()caffe('set_phase','train');
    caffe.set.phase.test     = @()caffe('set_phase','test');
    caffe.set.weights        = @(w)caffe('set_weights',w);
    caffe.set.input          = @(i)caffe('upload_input',i);
    %% GETTERS
    caffe.get.weights        = @()caffe('get_weights');
    caffe.get.output         = @()caffe('download_output');
    caffe.get.grads          = @()caffe('get_grads');
    %% ACTIONS
    caffe.action.init        = @()matcaffe_init(net.caffe.use_gpu,net.caffe.prototxt);
    caffe.action.reset       = @()caffe('reset');
    caffe.action.forward     = @()caffe('forward');
    caffe.action.backward    = @(diffs)caffe('backward',diffs);

end

