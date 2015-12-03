classdef TRAIN_SGD < handle
    %TRAIN_SGD Summary of this class goes here
    %   Detailed explanation goes here
    %% AUTHOR: PROVOS ALEXIS
    %   DATE:   20/5/2015
    %   FOR:    VISION TEAM - AUTH
    properties
        caffe   = [];
        Vt      = [];
        base_m  = [];
        base_lr = [];
        base_wd = [];
    end
    
    methods
        function obj = TRAIN_SGD(caffe)
            obj.caffe= caffe;
             for i=1:length(obj.caffe.net_structure.params)
                 for j=1:length(obj.caffe.net_structure.params(i).data)
                     obj.Vt(i).data{j}=zeros(size(obj.caffe.net_structure.params(i).data{j}));
                 end
             end            
        end
        function obj = set_learning_params(obj,base_m,base_lr,base_wd)          
            obj.base_m   = base_m;
            obj.base_lr  = base_lr;
            obj.base_wd  = base_wd;
        end
        
        function update_params(obj,sum_grads,bpi)   
            
            local_m      = obj.caffe.net_structure.m_mult  .* obj.base_m;            
            local_lr     = obj.caffe.net_structure.lr_mult .* obj.base_lr;
            local_wd     = obj.caffe.net_structure.wd_mult .* obj.base_wd;
            for i=1:length(sum_grads)
                for j=1:length(sum_grads(i).diff) %weight + bias / weights
                    
                    obj.Vt(i).data{j}=local_m(i,j) * obj.Vt(i).data{j} - (local_lr(i,j)/bpi) * sum_grads(i).diff{j} - local_wd(i,j)*local_lr(i,j)*obj.caffe.net_structure.params(i).data{j};
                    obj.caffe.net_structure.params(i).data{j}=obj.caffe.net_structure.params(i).data{j}+obj.Vt(i).data{j};
                end
            end
            %% SET PARAMS
%            obj.caffe.set.params();
        end
    end
    
end

