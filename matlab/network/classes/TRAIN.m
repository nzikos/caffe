classdef TRAIN < handle 
    %TRAIN Class used to store training parameters such as training
    %method,iterations,epochs,errors,target error
    
    properties
        error        =[];        
        method       =[];
        objects      =[];
        objects_pool =[];        
        iter         =[];
        epoch        =[];
        caffe        =[];
        
        batches_per_iter =[];
        
        training_fig = figure('name','training error');
    end
    
    methods
        %% INIT
        function train = TRAIN(arg_caffe,arg_objects)
            train.caffe        = arg_caffe;
            train.objects      = vectorize_objects_fpaths(arg_objects);
            train.objects_pool = [];
            train.error(1)     = 1;
        end
        %% SETTERS
        function set_method(train,method,options)
            switch(method)
                case 'sgd'
                    train.method = TRAIN_SGD(train.caffe,options);
                otherwise
                    APP_LOG('error_last',0,'Training method %s not supported',method);
            end            
        end
        
        function set_batches_per_iter(train,value)
            train.batches_per_iter=value;
        end
        %% FUNCTIONS
        function do_train(train)
                %refill
                if (length(train.objects_pool)<train.batches_per_iter*train.caffe.batch_size)
                    train.objects_pool =train.objects;
                end

                grads       =train.caffe.zero_struct;
                current_mse =0;
                for j=1:train.batches_per_iter
                    [batch,train.objects_pool]=create_random_batch(train.objects_pool,...
                                                                   train.caffe.batch_size);
                    this_grads =train.caffe.action.training_iter(batch);
                    scores     =train.caffe.get.output();
                    
                    this_mse   =mean(sum((batch{2}-scores{1}).^2,3),4);              
                    current_mse = current_mse + this_mse;
                    %RETRIEVE AND SUM NEW WEIGHT-BIAS DIFFS
                    for k=1:train.caffe.n_layers
                        for l=1:2
                            grads(k).weights{l}=grads(k).weights{l}+this_grads(k).weights{l};
                        end                    
                    end
                end
                %% Get mean of grads
                for j=1:train.caffe.n_layers
                    for k=1:2
                        grads(j).weights{k}=grads(j).weights{k}/train.batches_per_iter;
                    end
                end
                train.handle_error(current_mse);
                %% UPDATE WEIGHTS
                train.method.update_weights(grads);
        end
        
        function handle_error(train,current_mse)
            train.error(end+1) = current_mse/train.batches_per_iter;

            if isnan(train.error(end))
                thats_very_bad=1; %insert breakpoint for debug
            end          
            figure(train.training_fig);
            plot(train.error);            
        end
    end
    
end

