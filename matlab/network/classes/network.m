classdef network < handle
    %NETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        input_dims           =[];
        caffe                =[];
        train                =[];
        validation           =[];
        sets                 =[];

        iter                 =[];
        epoch                =[];

        iters_per_epoch      =[];
        iters_per_val        =[];
        
        max_epochs           =[];
        target               =[];
    end
    
    methods
        %% INIT
        function net = network(extraction_model)
            if strcmp(class(extraction_model),'extraction_model')
                
                net.caffe              = CAFFE();
                
                set                    = extraction_model.sets.set;
                net.train              = TRAIN(net.caffe,extraction_model.objects.data.(set{1}));
                net.validation         = VALIDATION(net.caffe,extraction_model.objects.data.(set{2}));

                net.iter               = 0;
                net.epoch              = 0;
                
                net.input_dims         = extraction_model.objects.dims;
            else
                APP_LOG('error_last',0,'Expected class "extraction_model"');
            end
        end
        %% SETTERS
        function net = set_batches_per_iter(net,batches_per_iter)

            net.train.set_batches_per_iter(batches_per_iter);
            net.iters_per_epoch  = floor(length(net.train.objects)/(batches_per_iter*net.caffe.batch_size));
        end
        
        function net = set_validations_per_epoch(net,vals_per_epoch)
            net.iters_per_val    = round(net.iters_per_epoch/vals_per_epoch);
        end
        
        function net = set_max_epochs(net,max_epochs)
            net.max_epochs = max_epochs;
        end
        
        function net = set_target(net,target_error)
            net.target = target_error;
        end                
        %% TRAIN/VAL
        function net = start(net)
            APP_LOG('header',0,'Training');
            net.caffe.set.phase.train();
            while (net.epoch < net.max_epochs) || (net.validation.error(end) > net.target)
                
                net.train.do_train();
                net.iter=net.iter+1;

                APP_LOG('info',0,'iter: %d/%d | epoch %d/%d | error: %1.15f',net.iter,net.iters_per_epoch,net.epoch,net.max_epochs,net.train.error(end));

                if ~mod(net.iter,net.iters_per_val) || net.iter==net.iters_per_epoch
                    APP_LOG('header',0,'Validating');
                    net.caffe.set.phase.test();
                    net.validation.do_validation();                    
                    APP_LOG('header',0,'Training');
                    net.caffe.set.phase.train();                    
                end

                if net.iter>=net.iters_per_epoch
                    net.iter=0;
                    net.epoch=net.epoch+1;
                end
            end
        end
    end
end
