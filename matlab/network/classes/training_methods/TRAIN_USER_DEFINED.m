classdef TRAIN_USER_DEFINED < handle
    %TRAIN_USER_DEFINED This class is a template of the user defined method
    %which will be used to update the weights of the cnn.
    
    %   Arguments of this class are the caffe handle class and
    %   user_parameters. User_parameters can be anything a user may want to
    %   read or modify through this class. If changes are going to be made 
    %   in user_parameters, in order to be noticed from the other classes,
    %   make sure you pass handlers (which are equivallent to pointers in
    %   C/C++).
    
    %%  AUTHOR: PROVOS ALEXIS
    %   DATE:   10/8/2015
    %   FOR:    VISION TEAM - AUTH
    
    properties (SetAccess = private)
        name    = 'MY sgd (MY stohastic gradient descent)'
        caffe   = [];
        Vt      = [];        
        lr      = [];
        m       = [];
        gamma   = [];
        wd      = [];
        validation;
        exit;
        failed_vals = [];
        failed_vals_threshold= [];        
        failed_lr_changes = [];
        failed_lr_changes_threshold = [];
        curr_val_idx;
    end
    
    methods
        function obj = TRAIN_USER_DEFINED(caffe,user_parameters)
            obj.caffe                      = caffe;
            obj.validation                 = user_parameters{1};
            obj.exit                       = user_parameters{2};
            obj.failed_vals                = 0;
            obj.failed_lr_changes          = 0;
            obj.failed_vals_threshold      = 0;
            obj.failed_lr_changes_threshold= 0;
            obj.curr_val_idx               = 1;
        end
        
        function init(obj)
             for i=1:obj.caffe.structure.n_layers
                 for j=1:2
                     obj.Vt(i).weights{j}=zeros(size(obj.caffe.weights(i).weights{j}));
                 end
             end
        end
        
        function set_params(obj,options)          
            switch (length(options))
                case 3 %lr
                    obj.lr      = options{1};
                    obj.m       = 0;   
                    obj.gamma   = 1;
                    obj.wd      = 0;             
                    obj.failed_vals_threshold=options{2};
                    obj.failed_lr_changes_threshold=options{3};
                case 4 %lr-m
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = 1;
                    obj.wd      = 0;
                    obj.failed_vals_threshold=options{3};
                    obj.failed_lr_changes_threshold=options{4};
                case 5 %lr-m-gamma
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = options{3};
                    obj.wd      = 0;
                    obj.failed_vals_threshold=options{4};
                    obj.failed_lr_changes_threshold=options{5};
                case 6 %lr-m-gamma-weight decay
                    obj.lr      = options{1};
                    obj.m       = options{2};
                    obj.gamma   = options{3};
                    obj.wd      = options{4};
                    obj.failed_vals_threshold=options{5};
                    obj.failed_lr_changes_threshold=options{6};
                otherwise
                    APP_LOG('last_error','Erroneous options passed to SGD method');
            end
        end
        
        function update_weights(obj,const_layers,sum_grads,bpi)
            if obj.curr_val_idx < length(obj.validation.average)
                obj.curr_val_idx=obj.curr_val_idx+1;
                if ~obj.validation.found_new_best
                    obj.failed_vals=obj.failed_vals+1;
                    APP_LOG('info','Failed validations %d/%d',obj.failed_vals,obj.failed_vals_threshold);
                    if(obj.failed_vals>=obj.failed_vals_threshold)
                        APP_LOG('info','SGD changes learning rate from %f to %f',obj.lr,obj.lr*obj.gamma);
                        obj.lr=obj.lr*obj.gamma;
                        obj.failed_vals=0;
                        obj.failed_lr_changes=obj.failed_lr_changes+1;
                        if(obj.failed_lr_changes>obj.failed_lr_changes_threshold)
                            APP_LOG('info','SGD raised the exit flag');
                            obj.exit.raise_flag();
                        end
                    end
                else
                    obj.failed_vals=0;
                    obj.failed_lr_changes=0;
                end
            end
%             wcoeff= (1-obj.lr*obj.wd);
%             for i=1:obj.caffe.n_layers
%                 if isempty(find(const_layers==i,1))
%                     for j=1:2
%                         obj.Vt(i).weights{j}=obj.m * obj.Vt(i).weights{j} - (obj.lr/bpi) * sum_grads(i).blob{j};
%                         obj.caffe.weights(i).weights{j}=wcoeff*obj.caffe.weights(i).weights{j}+obj.Vt(i).weights{j};
%                     end
%                 end
%             end
            lr_mult = obj.caffe.structure.lr_mult;
            for i=1:obj.caffe.structure.n_layers
                if isempty(find(const_layers==i,1))
                    for j=1:length(obj.caffe.weights(i).weights) %weight + bias / weights
                        %local_lr = lr * local_lr_mult;
                        %Vt = m*Vt - (local_lr/bpi)*grads - local_lr*wd*Wt;
                        local_lr = obj.lr * lr_mult{i,j};
                        obj.Vt(i).weights{j}=obj.m * obj.Vt(i).weights{j} - (local_lr/bpi) * sum_grads(i).blob{j} - obj.wd*local_lr*obj.caffe.weights(i).weights{j};
                        obj.caffe.weights(i).weights{j}=obj.caffe.weights(i).weights{j}+obj.Vt(i).weights{j};
                    end
                end
            end
            clear grads;
            obj.caffe.set.weights();
        end
    end
    
end

