classdef CAFFE < handle
    %CAFFE This class is the network's API with caffe API. Whenever Caffe
    %is needed to take an action, this action goes through this class.
    %
    %   Properties description
    %
    %%  set     : Contains all the set functions
    %
    %   set.phase.train -> Change caffe phase to train
    %   set.phase.test  -> Change caffe phase to test
    %   set.weights     -> Sets the weights from this class to the caffe 
    %                      GPU or CPU memory. Depends where caffe is running.
    %   set.input       -> Sends a batch of images with their labels to caffe.
    %   set.device      -> In case of GPU usage and multiple GPU system 
    %                      allows caffe to choose which GPU to use (default: 0)
    %
    %%  get     : Contains all the get functions
    %
    %   get.weights        -> Retrieve the weights from caffe.
    %   get.output         -> Retrieve the output from caffe. Depends upon 
    %                         network structure.
    %   get.grads          -> Retrieve grads computed through a backward 
    %                         propagation of errors.
    %   get.is_initialized -> Returns 1 if already initialized 0 otherwise.
    %
    %%  action  : Contains all the actions functions
    %
    %   action.reset    -> Release resources
    %   action.init     -> Initialize caffe. A path to the prototxt file 
    %                      containing the network structure should be 
    %                      passed.
    %   action.forward  -> perform a forward propagation of a batch.
    %   action.backward -> perform a backward propagation of errors 
    %                      computed by the forward propagation, computing grads.
    %
    %   action.forward_backward -> short command for the above two
    %   action.training_iter -> short command to upload input, do a
    %                           forward pass, a backward pass and return grads. 
    %                           Takes as input a batch.
    %
    % use_gpu       : Used to initialize caffe to use a CUDA enabled gpu or
    %                 not.
    %
    % prototxt      : The path to the prototxt file containing the network
    %                 structure.
    %
    % n_layers      : The number of layers where weights-bias exist.
    %       
    % zero_struct   : A dummy struct of zeroes to quick initialize grads
    %                 every training iteration.
    %
    % weights       : The network weights as returned by caffe
    %
    % labels        : Array containing the labels that this network will
    %                 be able to classify. [Name, contestID]
    %
    % batch_factory : functions to create a batch per phase, batch_size,
    %                 network input dimensions, variable to use flipped
    %                 objects.
    %
    %%      AUTHOR: PROVOS ALEXIS
    %       DATE:   20/5/2015
    %       FOR:    VISION TEAM - AUTH
    
    properties (SetAccess = private)
        set          = [];
        get          = [];
        action       = [];
        use_gpu      = [];
        prototxt     = [];
        n_layers     = [];
        zero_struct  = [];
        batch_factory= [];
        labels       = [];
    end
    properties (SetAccess = public)
        weights      = [];
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
            obj.action.init             = @()caffe('init',obj.prototxt,'train');
            obj.action.forward          = @()caffe('forward');
            obj.action.forward_backward = @()caffe('forward_backward');
            obj.action.backward         = @(diffs)caffe('backward',diffs);
            obj.action.training_iter    = @(batch)caffe('training_iter',batch);
            
            obj.batch_factory = BATCH_FACTORY();
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
        
       function set_labels(obj,arg_dataset)
            for i=length(arg_dataset):-1:1
                obj.labels(i).name      = arg_dataset(i).labels.name;
                obj.labels(i).contestID = arg_dataset(i).labels.contestID;
            end
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
            if obj.get.is_initialized() == 1
                obj.action.reset();
            end
            
            if ~exist(obj.prototxt,'file')
                % NOTE: you'll have to get network definition
                APP_LOG('last_error',0,'Could not find the prototxt. Do you have read/write permissions?');
            else
                % init network in train phase
                obj.action.init();
            end
            
            % set to use GPU or CPU
            if obj.use_gpu
                caffe('set_mode_gpu');
            else
                caffe('set_mode_cpu');
            end
            
            if isempty(obj.weights)
                obj.weights  = obj.get.weights();
            else
                obj.set.weights();
            end
            obj.n_layers = length(obj.weights);
            if isempty(obj.zero_struct)
                for i=1:obj.n_layers
                    for j=1:2
                        obj.zero_struct(i).weights{j}=zeros(size(obj.weights(i).weights{j}));
                    end
                end
            end
        end
	end
end