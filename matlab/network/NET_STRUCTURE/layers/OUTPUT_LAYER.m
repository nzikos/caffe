classdef OUTPUT_LAYER
    %OUTPUT_LAYER used to create object handlers for output layers whether 
    %they are error layers or link functions or nothing
    %
    %   This class is part of the NET_STRUCTURE() class.
    %
    %   Constructor is expecting a cell input in the form
    %   {
    %       error_function,
    %       optional_params
    %   }    
    %   Example inputs:
    %       {'SoftmaxWithLoss'}
    %       {'HingeLoss','L2'}
    %
    %   Error functions inherited from Caffe are
    %   1. SoftmaxWithLoss
    %   2. SigmoidCrossEntropyLoss
    %   3. EuclideanLoss
    %   4. HingeLoss
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   08/11/2015
    %  FOR:    VISION TEAM - AUTH    
    properties
        bottom;
        bottom_size;
        labels_size;
        top;
        top_size_train;
        top_size_test;
        top_size;
        name;
        
        type_train;
        type_test;
        
        params;
    end
    
    methods
        function obj = OUTPUT_LAYER(bottom,bottom_size,top,name,hyper_params)
            %% PROCESS LAYER'S CONNECTIONS
            obj.bottom            = bottom;
            obj.bottom_size       = bottom_size;
            obj.top               = top;
            obj.name              = name;
            
            %% PROCESS LAYER'S HYPER-PARAMETERS
            if isempty(hyper_params)
                APP_LOG('last_error','Output layer expects at least one parameter');
            end
            
            obj.type_train        = hyper_params{1};
            
            switch obj.type_train
                case 'SoftmaxWithLoss'
                    obj.type_test = 'Softmax';
                    obj.labels_size = [1 1 1]; %HxWxD
                case 'SigmoidCrossEntropyLoss'
                    obj.type_test = 'Sigmoid';
                    obj.labels_size = [1 1 obj.bottom_size(3)]; %HxWxD
                case 'EuclideanLoss'
                    obj.type_test = [];
                    obj.labels_size = [1 1 obj.bottom_size(3)]; %HxWxD
                case 'HingeLoss'
                    if length(hyper_params)~=2
                        obj.params = 'L1';
                        APP_LOG('warning','%s: HingeLoss parameter set to default L1',obj.name);
                    else
                        switch hyper_params{2}
                            case 'L1'
                                obj.params = hyper_params{2};
                            case 'L2'
                                obj.params = hyper_params{2};
                            otherwise
                                APP_LOG('%s: Unsupported parameter %s for HingeLoss layer. Use L1 or L2',obj.name,hyper_params{1});
                        end
                    end
                    obj.type_test = [];
                    obj.labels_size = [1 1 1]; %HxWxD
                otherwise
                    APP_LOG('last_error','Init %s: Unknown type %s.',obj.name,obj.type_train);
            end                    
            
            %% PROCESS LAYER'S OUTPUT SIZE
            obj.top_size_train       = [1,1,1];
            obj.top_size_test        = [1,1,obj.bottom_size(3)];
            obj.top_size             = obj.top_size_test;       %dummy var, needs fix
            
            %% PRINT LAYER'S STATUS -ON SCREEN/LOGS
            APP_LOG('debug','Layer %s:'                     ,obj.name);
            APP_LOG('debug','bottom: %s'                    ,obj.bottom);
            APP_LOG('debug','bottom_size: [%d, %d, %d]'     ,obj.bottom_size);
            APP_LOG('debug','top: %s'                       ,obj.top);
            APP_LOG('debug','top_size_train: [%d, %d, %d]'  ,obj.top_size_train);
            APP_LOG('debug','type_train: %s'                ,obj.type_train);
            switch obj.type_train
                case 'HingeLoss'
                    APP_LOG('debug','norm: %s',obj.params);
            end            
            APP_LOG('debug','top_size_test: [%d, %d, %d]'   ,obj.top_size_test);
            APP_LOG('debug','type_test: %s'                 ,obj.type_test);
            APP_LOG('debug','');                                
        end
        %% TRANSFORM LABELS FROM Labels.uid to ground truth vector CAFFE's error function expects
        function out = transform_labels(obj,uids)
            switch obj.type_train
                case 'SoftmaxWithLoss'
                    out = zeros(1,1,1,size(uids,1));
                    if(size(uids,2)~=1)
                        APP_LOG('last_error','%s: Is not used for multiclass classification problems',obj.type_train);
                    end
                    for i=1:size(uids,1)
                        out(1,1,1,i)=uids(i);
                    end
                case 'HingeLoss'
                    out = zeros(1,1,1,size(uids,1));
                    if(size(uids,2)~=1)
                        APP_LOG('last_error','%s: Is not used for multiclass classification problems',obj.type_train);
                    end
                    for i=1:size(uids,1)
                        out(1,1,1,i)=uids(i);
                    end
                case 'SigmoidCrossEntropyLoss'
                    out = zeros(1,1,obj.bottom_size(3),size(uids,1));
                    uids  = uids+1;                    
                    for i=1:size(uids,1)
                        out(1,1,uids(i,:),i)=1;
                    end
                case 'EuclideanLoss'
                    out = zeros(1,1,obj.bottom_size(3),size(uids,1));
                    uids  = uids+1;                    
                    for i=1:size(uids,1)
                        out(1,1,uids(i),i)=1;
                    end
                otherwise
                    APP_LOG('last_error','Transform labels %s: Unknown type %s.',obj.name,obj.type_train);                    
            end
            out=single(out);
        end
        %% COMPUTE ERROR OF EACH SAMPLE GIVEN THE GROUND-TRUTH + SCORES + error function
        function [error,uids] = compute_error(obj,scores,ground_truth)
            switch obj.type_train
                case 'SoftmaxWithLoss'
                    %Multinomial logistic Loss
                    uids               = ground_truth+1; % caffe is zero-based
                    %scores(scores<realmin('single'))=realmin('single');
                    scores(scores<eps) = eps;
                    for i=size(scores,2):-1:1
                        error(i,1) = -log(scores(uids(i),i));
                    end
                case 'HingeLoss'
                    uids                = ground_truth+1;
                    for i=size(scores,2):-1:1
                        signs = -ones(1,size(scores,1));
                        signs(uids(i))=1;
                        switch obj.params
                            case 'L1'
                                error(i,1) = sum(max(0,1-signs.*scores(:,i)));
                            case 'L2'
                                error(i,1) = sum(max(0,1-signs.*scores(:,i)).^2);
                            otherwise
%                               APP_LOG('%s: Unsupported parameter %s for HingeLoss layer. Use L1 or L2',obj.name,hyper_params{1});
                        end
                    end
                case 'SigmoidCrossEntropyLoss'
                    scores(scores<eps) = eps;
                    for i=size(scores,2):-1:1
                        uids(i,:) = find(ground_truth(:,i)==1);                        
                        error(i,1) = sum(ground_truth(:,i).*log(scores(:,i)) + (1-ground_truth(:,i)).*log(1-scores(:,i)));
                    end
                case 'EuclideanLoss'
                    for i=size(ground_truth,2):-1:1
                        uids(i,:) = find(ground_truth(:,i)==1);
                    end
                    error(:,1) = sum((ground_truth-scores).^2,1);
                otherwise
                    APP_LOG('last_error','Compute error %s: Unknown type %s.',obj.name,obj.type_train);
            end
        end
        %% PRINT LAYER ON PROTOTXT
        function print_layer(obj,varargin)
            switch varargin{2}
                case 'train'
                    formatSpec = ['layer {\n\t'...
                                     'name: "%s"\n\t'...
                                     'type: "%s"\n\t'...
                                     'bottom: "%s"\n\t'...
                                     'bottom: "label"\n\t'...
                                     'top: "%s"\n\t'...
                                     '%s'...
                                 '}\n'];
                    print_params=[];
                    switch obj.type_train
                        case 'HingeLoss'
                            print_params = sprintf('hinge_loss_param {\n\t\tnorm: %s\n\t}\n',obj.params);
                    end
                    if ~isempty(varargin{1})
                        fprintf(varargin{1},formatSpec,obj.name,obj.type_train,obj.bottom,obj.top,print_params);
                    else
                        APP_LOG('warning','No filepath to print prototxt supplied');
                    end
                case 'validation'
                    if ~isempty(obj.type_test)
                        formatSpec = ['layer {\n\t'...
                                         'name: "%s"\n\t'...
                                         'type: "%s"\n\t'...
                                         'bottom: "%s"\n\t'...
                                         'top: "%s"\n\t'...
                                         '%s'...
                                      '}\n'];
                        print_params=[];
                        if ~isempty(varargin{1})
                            fprintf(varargin{1},formatSpec,obj.name,obj.type_test,obj.bottom,obj.top,print_params);
                        else
                            APP_LOG('warning','No filepath to print prototxt supplied');
                        end
                    end
                case 'test'
                    if ~isempty(obj.type_test)
                        formatSpec = ['layer {\n\t'...
                                         'name: "%s"\n\t'...
                                         'type: "%s"\n\t'...
                                         'bottom: "%s"\n\t'...
                                         'top: "%s"\n\t'...
                                         '%s'...
                                      '}\n'];
                        print_params=[];
                        if ~isempty(varargin{1})
                            fprintf(varargin{1},formatSpec,obj.name,obj.type_test,obj.bottom,obj.top,print_params);
                        else
                            APP_LOG('warning','No filepath to print prototxt supplied');
                        end
                    end
                otherwise
                    APP_LOG('last_error','%s: Unknown Phase, use train/validation/test',obj.name);
            end
        end
    end        
end

