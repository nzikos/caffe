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
        function init(obj)
             for i=1:length(obj.caffe.net_structure.params)
                 for j=1:length(obj.caffe.net_structure.params(i).data)
                     obj.Vt(i).data{j}=zeros(size(obj.caffe.net_structure.params(i).data{j}));
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
            
            local_lr     = obj.caffe.net_structure.lr_mult .* obj.lr;
            local_wd     = obj.caffe.net_structure.wd_mult .* obj.wd;
            local_m      = obj.caffe.net_structure.m_mult  .* obj.m;
            for i=1:length(sum_grads)
                for j=1:length(obj.caffe.net_structure.params(i).data) %weight + bias / weights
                    
                    obj.Vt(i).data{j}=local_m(i,j) * obj.Vt(i).data{j} - (local_lr(i,j)/bpi) * sum_grads(i).diff{j} - local_wd(i,j)*local_lr(i,j)*obj.caffe.net_structure.params(i).data{j};
                    obj.caffe.net_structure.params(i).data{j}=obj.caffe.net_structure.params(i).data{j}+obj.Vt(i).data{j};
                end
            end
            clear grads;
            obj.caffe.set.params();
            obj.curr_steps=obj.curr_steps+1;
        end
    end
    
end

