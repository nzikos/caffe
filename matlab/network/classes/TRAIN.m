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
        training_fig     =figure('name','training error');
        const_layers     =[];
    end
    
    methods
        %% INIT
        function train = TRAIN(arg_caffe,arg_objects)
            train.caffe        = arg_caffe;
            train.objects      = vectorize_objects_fpaths(arg_objects);
            train.objects_pool = [];
            train.error(1)     = 1;
            
            %figure initialization
            %set(train.training_fig,'MenuBar','none');
        end
        %% SETTERS
        function set_method(train,method)
            switch(method)
                case 'sgd'
                    train.method = TRAIN_SGD(train.caffe);
                otherwise
                    APP_LOG('error_last',0,'Training method %s not supported',method);
            end            
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
                %refill
                if (length(train.objects_pool)<train.batches_per_iter*rcaffe.batch_factory.batch_size)
                    train.objects_pool =train.objects;
                end

                grads       =rcaffe.zero_struct;
                current_mse =0;
                for j=1:train.batches_per_iter
                    
                    [batch,train.objects_pool]=rcaffe.batch_factory.create_training_batch(train.objects_pool);
                    
                    this_grads =rcaffe.action.training_iter(batch(1:2));
                    scores     =rcaffe.get.output();
                    this_mse   =mean(sum((batch{3}-scores{1}).^2,3),4);              
                    %this_mse   =scores{2}; % <-- loss layer output                    
                    current_mse = current_mse + this_mse;
                    %RETRIEVE AND SUM NEW WEIGHT-BIAS DIFFS
                    for k=1:rcaffe.n_layers
                        for l=1:2
                            grads(k).weights{l}=grads(k).weights{l}+this_grads(k).weights{l};
                        end                    
                    end
                end
                %% Get mean of grads
                for j=1:rcaffe.n_layers
                    for k=1:2
                        grads(j).weights{k}=grads(j).weights{k}/train.batches_per_iter;
                    end
                end
                train.handle_error(current_mse);
                %% UPDATE WEIGHTS
                train.method.update_weights(train.const_layers,grads);
        end
        
        function handle_error(train,current_mse)
            train.error(end+1) = current_mse/train.batches_per_iter;

            figure(train.training_fig);
            plot(train.error);            
        end
    end
    
end

