classdef TRAIN < handle 
%TRAIN Class used to store training parameters-statistics and execute 
%training iterations.
%
%% Method description
%
%  TRAIN(caffe,batch_factory)     : used to initialize an object of 
%                                   this class passing caffe and batch_factory
%                                   handler.
%  set_method(method,arguments)   : Set the method responsible for
%                                   weight update. More info inside each
%                                   method function.
%  set_batches_per_iter(value)    : Set how many batches should pass
%                                   before weight update method is
%                                   called. Gradients are the mean value of
%                                   passed batches.
%  do_train()                     : Perform a full iteration of x
%                                   batches and sum their computed
%                                   gradients. Then call the method
%                                   which updates the weights by
%                                   supplying the gradients, the constant
%                                   layers and the number of passed batches.
%% Properties description
%
%  error                         : Array to hold the error produced from 1 
%                                  iteration IF train.fetch_train_error
%                                  is enabled.
%  method                        : Handler to store the object which performs
%                                  the weight updates.
%  caffe                         : Handler to communicate with caffe
%  batch_factory                 : Handler to communicate with batch_factory
%  batches_per_iter              : The value indicating how many passed
%                                   batches are called an iteration.
%  fetch_train_error             : boolean value to fetch or not the training
%                                  error per iteration. Setting to false may
%			                       reduce training time. (default: 1)
%% AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH
properties
    error;
    method;
    caffe;
    batch_factory;
    batches_per_iter  = 1;
    fetch_train_error = 1;
end

methods
    %% INIT
    function train = TRAIN(arg_caffe,arg_batch_factory)
        train.caffe              = arg_caffe;
        train.batch_factory      = arg_batch_factory;
        train.error(1)           = inf;
    end
    %% SETTERS
    function set(train,method,args)
        method = upper(method);
        switch(method)
            case 'SGD'
                train.method = TRAIN_SGD(train.caffe);
                train.method.set_learning_params(args{1},args{2},args{3});                
            case 'ADAGRAD'
                train.method = TRAIN_ADAGRAD(train.caffe);
                train.method.set_learning_params(args{1},args{2});                
            otherwise
                APP_LOG('last_error','Training method %s not supported',method);
        end
    end
    
    function set_batches_per_iter(train,value)
        train.batches_per_iter=value;
    end
    
    function fetch_error(train,value)
        train.fetch_train_error = value;
    end
    
    %% FUNCTIONS
    function sum_grads = do_train(train)
        rcaffe = train.caffe;
        bfactory = train.batch_factory;
        
        current_error =0;
        
        for i=1:train.batches_per_iter
            batch=bfactory.create_training_batch();
            switch i
                case 1
                    sum_grads = rcaffe.action.training_iter(batch);
                otherwise
                    this_grads= rcaffe.action.training_iter(batch);
                    for j=1:length(sum_grads)
                        for k=1:length(sum_grads(j).diff)
                            sum_grads(j).diff{k}=sum_grads(j).diff{k}+this_grads(j).diff{k};
                        end
                    end
                    clear this_grads;
            end
            
            if(train.fetch_train_error)
                loc_error     = rcaffe.get.output();
                current_error = current_error + loc_error{1}; % <-- loss layer output
            end
            
        end
        if (train.fetch_train_error)
            train.error(end+1)=current_error/train.batches_per_iter;
        end
        %% UPDATE PARAMS
        train.method.update_params(sum_grads,train.batches_per_iter);
        %% SET PARAMS
        rcaffe.set.params();
    end
end
end
