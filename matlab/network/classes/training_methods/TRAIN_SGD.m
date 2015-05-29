classdef TRAIN_SGD < handle
    %TRAIN_SGD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private)
        name    = 'sgd (stohastic gradient descent)'
        caffe   = [];
        lr      = [];
        m       = [];
        Vt      = [];
        weights = [];
    end
    
    methods
        function obj = TRAIN_SGD(caffe,options)
            obj.caffe   = caffe;
            obj.Vt      = caffe.zero_struct;
            obj.weights = caffe.get.weights();
            obj.lr      = options{1};
            obj.m       = options{2};
        end
        
        function update_weights(obj,grads)
            for i=1:obj.caffe.n_layers
                for j=1:2
                    obj.Vt(i).weights{j}=obj.m * obj.Vt(i).weights{j} - obj.lr * grads(i).weights{j};
                end
                for j=1:2
                    obj.weights(i).weights{j}=obj.weights(i).weights{j}+obj.Vt(i).weights{j};
                end
            end
            obj.caffe.set.weights(obj.weights);
        end
    end
    
end

