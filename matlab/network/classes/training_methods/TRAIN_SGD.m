classdef TRAIN_SGD < handle
    %TRAIN_SGD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private)
        name    = 'sgd (stohastic gradient descent)'
        caffe   = [];
        Vt      = [];        
        lr      = [];
        m       = [];
        gamma   = [];
        stepsize= [];
        wd      = [];
        curr_steps = [];
    end
    
    methods
        function obj = TRAIN_SGD(caffe)
            obj.caffe= caffe;
        end
        function obj = init(obj)
            for i=1:obj.caffe.structure.n_layers
                for j=1:length(obj.caffe.params(i).data) %weight + bias / weights
                    obj.Vt(i).data{j}=zeros(size(obj.caffe.params(i).data{j}));
                end
            end
            obj.curr_steps = 1;
        end
        function obj = set_learning_params(obj,options)          
            switch (length(options))
                case 1 %lr
                    obj.lr      = options{1};
                    obj.m       = 0;   
                    obj.gamma   = 1;
                    obj.stepsize= 1000000; %random
                    obj.wd      = 0;                                        
                case 2 %lr-m
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = 1;
                    obj.stepsize= 1000000; %random
                    obj.wd      = 0;                                        
                case 4 %lr-m-gamma
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = options{3};
                    obj.stepsize= options{4};
                    obj.wd      = 0;                    
                case 5 %lr-m-gamma-weight decay
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = options{3};
                    obj.stepsize= options{4};
                    obj.wd      = options{5};
                otherwise
                    APP_LOG('last_error','Erroneous options passed to SGD method');
            end
        end
        
        function update_params(obj,sum_grads,bpi)
            if ~mod(obj.curr_steps+1,obj.stepsize)
                APP_LOG('info','SGD changes learning rate from %f to %f',obj.lr,obj.lr*obj.gamma);
%               APP_LOG('info','SGD changes momentum rate from %f to %f',obj.m,obj.m*obj.gamma + 1 - obj.gamma);
                obj.lr=obj.lr*obj.gamma;
%               obj.m =obj.m*obj.gamma + 1 - obj.gamma;
            end
            
            lr_mult      = obj.caffe.structure.lr_mult;
            n_layers     = obj.caffe.structure.n_layers;
            for i=1:n_layers
                for j=1:length(obj.caffe.params(i).data) %weight + bias / weights
                    %local_lr = lr * local_lr_mult;
                    %Vt = m*Vt - (local_lr/bpi)*grads - local_lr*wd*Wt;
                    local_lr = obj.lr * lr_mult{i,j};
                    obj.Vt(i).data{j}=obj.m * obj.Vt(i).data{j} - (local_lr/bpi) * sum_grads(i).diff{j} - obj.wd*local_lr*obj.caffe.params(i).data{j};
                    obj.caffe.params(i).data{j}=obj.caffe.params(i).data{j}+obj.Vt(i).data{j};
                end
            end
            clear grads;
            obj.caffe.set.params();
            obj.curr_steps=obj.curr_steps+1;
        end
    end
    
end

