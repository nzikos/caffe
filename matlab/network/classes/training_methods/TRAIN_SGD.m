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
        function obj = TRAIN_SGD(caffe,options)
            obj.caffe      = caffe;
            obj.Vt         = caffe.zero_struct;
            obj.curr_steps = 1;
            
            switch (length(options))
                case 1
                    obj.lr      = options{1};
                    obj.m       = 0;   
                    obj.gamma   = 1;
                    obj.stepsize= 1000000; %random
                    obj.wd      = 0;                                        
                case 2
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = 1;
                    obj.stepsize= 1000000; %random
                    obj.wd      = 0;                                        
                case 4
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = options{3};
                    obj.stepsize= options{4};
                    obj.wd      = 0;                    
                case 5
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = options{3};
                    obj.stepsize= options{4};
                    obj.wd      = options{5};
                otherwise
                    APP_LOG('last_error',0,'Erroneous options passed to SGD method');
            end
        end
        
        function update_weights(obj,const_layers,grads)
            
            if ~mod(obj.curr_steps,obj.stepsize)
                APP_LOG('info',3,'SGD changes learning rate from %f to %f',obj.lr,obj.lr*obj.gamma);
                obj.lr=obj.lr*obj.gamma;
                obj.wd=obj.wd*obj.gamma;
            end
            
            wcoeff= (1-obj.lr*obj.wd);
            for i=1:obj.caffe.n_layers
                
                if isempty(find(const_layers==i))
                    for j=1:2
                        obj.Vt(i).weights{j}=obj.m * obj.Vt(i).weights{j} - obj.lr * grads(i).weights{j};
                    end
                
                    for j=1:2
                        obj.caffe.weights(i).weights{j}=wcoeff*obj.caffe.weights(i).weights{j}+obj.Vt(i).weights{j};
                    end
                end
                
            end
            
            obj.caffe.set.weights();
            obj.curr_steps=obj.curr_steps+1;
        end
    end
    
end

