classdef TRAIN_ADAGRAD < handle
    %TRAIN_ADAGRAD Inherited from CAFFE
    %
    %% AUTHOR: PROVOS ALEXIS
    %   DATE:   20/5/2015
    %   FOR:    VISION TEAM - AUTH
    properties
        caffe            = [];
        sum_grads_pow2   = [];
        base_lr          = [];
        eps              = [];
    end
    
    methods
        function obj = TRAIN_ADAGRAD(caffe)
            obj.caffe= caffe;
             for i=1:length(obj.caffe.net_structure.params)
                 for j=1:length(obj.caffe.net_structure.params(i).data)
                     obj.sum_grads_pow2(i).data{j}=zeros(size(obj.caffe.net_structure.params(i).data{j}));
                 end
             end
        end
        
        function obj = set_learning_params(obj,base_lr,epsilon)
            obj.base_lr  = base_lr;
            obj.eps      = epsilon;
        end
        
        function update_params(obj,sum_grads,bpi)
            
            local_lr     = obj.caffe.net_structure.lr_mult .* obj.base_lr;

            for i=1:length(sum_grads)
                for j=1:length(sum_grads(i).diff) %weight + bias / weights
                                    
                    %Add grads to sum_grads_pow_2
                    obj.sum_grads_pow2(i).data{j}=obj.sum_grads_pow2(i).data{j} + sum_grads(i).diff{j}.^2;
                                        
                    obj.caffe.net_structure.params(i).data{j}=obj.caffe.net_structure.params(i).data{j} - (local_lr(i,j)/bpi) * (sum_grads(i).diff{j}./sqrt(obj.sum_grads_pow2(i).data{j}+obj.eps));
                end
            end
            %% SET PARAMS
%            obj.caffe.set.params();
        end
    end
    
end

