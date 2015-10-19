classdef TRAIN < handle 
%TRAIN Class used to store training parameters-statistics and execute 
%training iterations.
%
%% Method description
%
%  TRAIN(caffe)                   : used to initialize an object of 
%                                   this class passing caffe handler.
%  set_method(method,user_params) : Set the method responsible for
%                                   weight update. More info inside
%                                   method function
%  set_batches_per_iter(value)    : Set how many batches should pass
%                                   before weight update method is
%                                   called. Grads are the mean value of
%                                   the passed batches.
%  set_constant_layers(value)     : Set which layers weights should not
%                                   be updated
%  do_train()                     : Perform a full iteration of x
%                                   batches and sum their outputed
%                                   gradients. Then call the method
%                                   which updates the weights by
%                                   supplying the gradients, the constant
%                                   layers and the number of passed batches.
%% Properties description
%
%  error                          : Array to hold the error produced
%                                   from 1 iteration IF
%                                   caffe.compute_train_error is
%                                   enabled.
%  method                         : Handler to store the object which
%                                   performs the weight updates.
%  caffe                          : Handler to communicate with caffee
%
%  batches_per_iter               : The value indicating how many passed
%                                   batches are called an iteration.
%  const_layers                   : Value or array of values indicating
%                                   which layers wont be updated.
% compute_train_error             : boolean value to compute or not the training
%                                   error per iteration. Setting to false may
%				    reduce training time. (default: 1)
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH
properties
    error               = [];
    method              = [];
    caffe               = [];
    
    batches_per_iter    = [];
    const_layers        = [];
    compute_train_error = 1;
end

methods
    %% INIT
    function train = TRAIN(arg_caffe)
        train.caffe              = arg_caffe;
        train.error(1)           = inf;
    end
    %% SETTERS
    function set_method(train,method,user_parameters)
        method = upper(method);
        switch(method)
            case 'SGD'
                train.method = TRAIN_SGD(train.caffe);
            case 'USER_DEFINED'
                train.method = TRAIN_USER_DEFINED(train.caffe,user_parameters);
            otherwise
                APP_LOG('last_error','Training method %s not supported',method);
        end
        train.method.init();
    end
    
    function set_batches_per_iter(train,value)
        train.batches_per_iter=value;
    end
    
    function set_constant_layers(train,value)
        train.const_layers = value;
    end
    
    %% FUNCTIONS
    function do_train(train)
        rcaffe = train.caffe;
        bfactory = rcaffe.batch_factory;
        
        current_error =0;

        %We are assuming that batch size * iters is smaller than number of objects
        if (length(bfactory.train_objects)-(bfactory.train_objects_pos-1)<rcaffe.structure.train_batch_size*train.batches_per_iter)
            APP_LOG('debug','Pool size found to contain %d objects',length(bfactory.train_objects)-(bfactory.train_objects_pos-1));
            APP_LOG('debug','Random rearrangement of training objects');
            rand_idxs = randperm(length(bfactory.train_objects),length(bfactory.train_objects));
            bfactory.train_objects      = bfactory.train_objects(rand_idxs);
            bfactory.train_objects_pos  = 1;
        end
        
        for i=1:train.batches_per_iter
            batch=bfactory.create_training_batch();
            
            switch i
                case 1
                    sum_grads = rcaffe.action.training_iter(batch);
                otherwise
                    this_grads= rcaffe.action.training_iter(batch);
                    for j=1:rcaffe.structure.n_layers
                        if isempty(find(train.const_layers==j,1))
                            for k=1:2
                                sum_grads(j).blob{k}=sum_grads(j).blob{k}+this_grads(j).blob{k};
                            end
                        end
                    end
                    clear this_grads;
            end
            
            if(train.compute_train_error)
                scores        = rcaffe.get.output();
                current_error = current_error + scores{1}; % <-- loss layer output
            end
            
        end
        if(train.compute_train_error)
            train.error(end+1)=current_error/train.batches_per_iter;
        end
        %% UPDATE WEIGHTS
        train.method.update_weights(train.const_layers,sum_grads,train.batches_per_iter);
    end
end
end
