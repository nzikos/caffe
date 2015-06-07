classdef CAFFE < handle
    %CAFFE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (SetAccess = private)
        set        = [];
        get        = [];
        action     = [];
        use_gpu    = [];
        prototxt   = [];
        n_layers   = [];
        batch_size = [];
        zero_struct= [];
    end
    properties (SetAccess = public)
        weights    = [];
    end
    methods
        %% INIT 
        function obj = CAFFE()
            %CAFFE SETTERS HANDLERS
            obj.set.phase.train         = @()caffe('set_phase','train');
            obj.set.phase.test          = @()caffe('set_phase','test');
            obj.set.weights             = @()caffe('set_weights',obj.weights);
            obj.set.input               = @(i)caffe('upload_input',i);
            obj.set.device              = @(id)caffe('set_device',id);
            %CAFFE GETTERS HANDLERS
            obj.get.weights             = @()caffe('get_weights');
            obj.get.output              = @()caffe('download_output');
            obj.get.grads               = @()caffe('get_grads');
            obj.get.is_initialized      = @()caffe('is_initialized');
            %CAFFE ACTIONS HANDLERS
            obj.action.reset            = @()caffe('reset');
            obj.action.forward          = @()caffe('forward');
            obj.action.forward_backward = @()caffe('forward_backward');
            obj.action.backward         = @(diffs)caffe('backward',diffs);
            obj.action.training_iter    = @(batch)caffe('training_iter',batch);
        end
        
        %% SETTERS
        function set_use_gpu(obj,arg_use_gpu)
            if arg_use_gpu==0 || arg_use_gpu==1
                obj.use_gpu = arg_use_gpu;
            else
                APP_LOG('last_error',0,'Valid inputs for use_gpu are 0 or 1');
            end
        end
        
        function set_net_structure(obj,arg_prototxt)
            if exist(arg_prototxt,'file')
                [~,~,ext] = fileparts(arg_prototxt);
                if strcmp(ext,'.prototxt')
                    obj.prototxt = arg_prototxt;
                else
                    APP_LOG('last_error',0,'Invalid extension %s. Expected .prototxt',ext);
                end
            else
                APP_LOG('last_error',0,'%s not found',arg_prototxt);
            end
        end
        
        function set_batch_size(obj,batch_size)
            obj.batch_size = batch_size;
        end
        
        function set_layer(obj,layer_id,init_weights,init_bias)
            if obj.get.is_initialized()
                obj.weights  = obj.get.weights();
                obj.weights(layer_id).weights{1}=single(init_weights);
                obj.weights(layer_id).weights{2}=single(repmat(init_bias,size(obj.weights(layer_id).weights{2})));

                obj.set.weights();
            else
                APP_LOG('last_error',0,'Initialize caffe before initiallizing a layer');
            end
        end
        
        %% FUNCTIONS
        function init(obj)
            matcaffe_init(obj.use_gpu,obj.prototxt);
            obj.weights  = obj.get.weights();
            obj.n_layers = length(obj.weights);
            for i=1:obj.n_layers
                for j=1:2
                    obj.zero_struct(i).weights{j}=zeros(size(obj.weights(i).weights{j}));
                end
            end
        end
	end
end