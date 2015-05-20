function caffe = init_caffe( caffe )
%% INITIALIZE CAFFE
    caffe.action.init();
    caffe.set.phase.train();
    %put weights into model
    caffe.weights               = caffe.get.weights();
    caffe.nlayers               = length(caffe.weights);
    %Allocate a dummy space to quick-set weight, grads, etc to zero
    caffe.zero_struct = caffe.weights;
    for i=1:caffe.nlayers
        caffe.zero_struct(i).weights{1} = single(zeros(size(caffe.weights(i).weights{1})));
        caffe.zero_struct(i).weights{2} = single(zeros(size(caffe.weights(i).weights{2})));
    end        

end

