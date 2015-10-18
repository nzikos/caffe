classdef CAFFE < handle
    %CAFFE This class is the network's wrapper with caffe API. Whenever Caffe
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
    %   get.output(x)      -> Retrieve the output of layer x from caffe. 
    %                         Depends upon network structure.
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
    % use_gpu             : Used to initialize caffe to use a CUDA enabled 
    %                       gpu or not.
    %
    % structure           : The network structure
    %       
    % weights             : The network weights as returned by caffe
    %
    % labels              : Array containing the labels that this network 
    %                       will be able to classify. [Name, contestID]
    %
    % batch_factory       : functions to create a batch per phase.
    %
    %% AUTHOR: PROVOS ALEXIS
    %  DATE:   20/5/2015
    %  FOR:    VISION TEAM - AUTH
    
    properties (SetAccess = private)
        set                 = [];
        get                 = [];
        action              = [];
        use_gpu             = [];
        current_phase       = [];
        structure           = [];
        batch_factory       = [];
        train_prototxt      = [];
        validation_prototxt = [];
        test_prototxt       = [];
    end
    properties (SetAccess = public)
        weights             = [];
        labels              = [];        
    end
    methods
        %% INIT 
        function obj = CAFFE(prototxt_path)
            %CAFFE SETTERS HANDLERS
            obj.set.weights             = @()caffe('set_weights',obj.weights);
            obj.set.input               = @(batch)caffe('upload_input',batch);
            obj.set.device              = @(id)caffe('set_device',id);
            %CAFFE GETTERS HANDLERS
            obj.get.weights             = @()caffe('get_weights');
            obj.get.output              = @()caffe('download_output');
            obj.get.grads               = @()caffe('get_grads');
            obj.get.is_initialized      = @()caffe('is_initialized');
            obj.get.blobs_number        = @()caffe('get_blobs_number');
            obj.get.blob                = @(i)caffe('get_blob',i);
            %CAFFE ACTIONS HANDLERS
            obj.action.reset            = @()caffe('reset');
            obj.action.init             = @()caffe('init',obj.train_prototxt,'train');
            obj.action.forward          = @()caffe('forward');
            obj.action.forward_backward = @()caffe('forward_backward');
            obj.action.backward         = @(diffs)caffe('backward',diffs);
            obj.action.training_iter    = @(batch)caffe('training_iter',batch);
            
            obj.structure               = NET_STRUCTURE(prototxt_path);            
            obj.batch_factory           = BATCH_FACTORY(obj.structure);
        end
        
        %% SETTERS
        function set_use_gpu(obj,arg_use_gpu)
            if arg_use_gpu==0 || arg_use_gpu==1
                obj.use_gpu = arg_use_gpu;
            else
                APP_LOG('last_error','Valid inputs for use_gpu are 0 or 1');
            end
        end

        function set_phase(obj,phase)
            obj.action.reset();
            switch phase
                case 'train'
                    if isempty(obj.train_prototxt)
                        obj.train_prototxt = obj.structure.create_prototxt('train');
                    end
                    caffe('init',obj.train_prototxt,'train');
                    obj.current_phase='train';
                case 'validation'
                    if isempty(obj.validation_prototxt)                    
                        obj.validation_prototxt = obj.structure.create_prototxt('validation');                    
                    end
                    caffe('init',obj.validation_prototxt,'test');
                    obj.current_phase='validation';                    
                case 'test'
                    if isempty(obj.test_prototxt)
                        obj.test_prototxt       = obj.structure.create_prototxt('test');
                    end
                    caffe('init',obj.test_prototxt,'test');
                    obj.current_phase='test';
                otherwise
                    APP_LOG('last_error','Invalid phase. Expected train/validation/test, got %s',phase);
            end
            if ~isempty(obj.weights)
                obj.set.weights();
            end
        end
        
       function set_labels(obj,arg_dataset)
            for i=length(arg_dataset):-1:1
                obj.labels(i).name      = arg_dataset(i).labels.name;
                obj.labels(i).contestID = arg_dataset(i).labels.contestID;
            end
        end
        function set_structure(obj,arg_structure)
            if strcmp(class(arg_structure),'NET_STRUCTURE')
                obj.structure = arg_structure;
            else
                APP_LOG('last_error','Expected object of class "NET_STRUCTURE"');
            end
        end
        function reset_object_input(obj,phase,object_input_size)
            switch(phase)
                case 'train'
                    obj.structure.set_batch_size(phase,object_input_size);
                    obj.structure.validate_structure();
                    obj.train_prototxt = obj.structure.create_prototxt(phase);                    
                case 'validation'
                    obj.structure.set_batch_size(phase,object_input_size);
                    obj.structure.validate_structure();
                    obj.validation_prototxt = obj.structure.create_prototxt(phase);                    
                case 'test'
                    obj.structure.set_batch_size(phase,object_input_size);
                    obj.structure.validate_structure();
                    obj.test_prototxt = obj.structure.create_prototxt(phase);
                otherwise
                    APP_LOG('last_error','Invalid phase "%s". Use train/validation/test instead',phase);
            end
            if strcmp(phase,obj.current_phase)
                obj.set_phase(phase);
            end
        end

        function set_layer(obj,layer_id,init_weights,init_bias)
            if obj.get.is_initialized()
                obj.weights  = obj.get.weights();
                obj.weights(layer_id).weights{1}=single(init_weights);
                obj.weights(layer_id).weights{2}=single(repmat(init_bias,size(obj.weights(layer_id).weights{2})));

                obj.set.weights();
            else
                APP_LOG('last_error','Initialize caffe before initiallizing a layer');
            end
        end
        
        %% FUNCTIONS
        function init(obj,phase)          
            if obj.get.is_initialized() == 1
                obj.action.reset();
            end            
            % Validate network structure
            obj.structure.validate_structure();
            obj.set_phase(phase);
            
            % set to use GPU or CPU
            if obj.use_gpu
                caffe('set_mode_gpu');
            else
                caffe('set_mode_cpu');
            end
            
            if isempty(obj.weights)                 %If no weights loaded / clean start scenario
                obj.weights  = obj.get.weights();
            else                                    %If net has loaded weights / restore a previous execution
                obj.set.weights();
            end
        end
	end
end
