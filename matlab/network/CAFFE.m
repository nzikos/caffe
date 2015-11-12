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
    %   set.params      -> Sets the weights and bias from this class to Caffe 
    %                      GPU or CPU memory. Depends where caffe is running.
    %   set.input       -> Sends a batch of images with their labels to caffe.
    %   set.device      -> In case of GPU usage and multiple GPU system 
    %                      allows caffe to choose which GPU to use (default: 0)
    %
    %%  get     : Contains all the get functions
    %
    %   get.params         -> Retrieve the weights and bias from caffe. (for debug)
    %   get.output         -> Retrieve the output of layer x from caffe.
    %                         Depends upon network structure (for train/val).
    %   get.gradients      -> Retrieve gradients computed through a backward 
    %                         propagation of errors. (for train)
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
    % net_structure       : The network's structure.
    %
    % labels              : Array containing the labels that this network 
    %                       will be able to classify. [Name, contestID]
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
        train_prototxt      = [];
        validation_prototxt = [];
        test_prototxt       = [];
    end
    properties (SetAccess = public)
        net_structure       = [];
        labels              = [];
    end
    methods
        %% INIT 
        function obj = CAFFE(net_struct)
            %CAFFE SETTERS HANDLERS
            obj.set.params              = @()caffe('set_params',obj.net_structure.params);
            obj.set.input               = @(batch)caffe('upload_input',batch);
            %CAFFE GETTERS HANDLERS
            obj.get.params              = @()caffe('get_params');
            obj.get.output              = @()caffe('download_output');
            obj.get.gradients           = @()caffe('get_gradients');
            obj.get.is_initialized      = @()caffe('is_initialized');
            obj.get.blobs_number        = @()caffe('get_blobs_number');
            obj.get.blob_data           = @(i)caffe('get_blob_data',i);
            obj.get.blob_diff           = @(i)caffe('get_blob_diff',i);
            %CAFFE ACTIONS HANDLERS
            obj.action.reset            = @()caffe('reset');
            obj.action.init             = @()caffe('init',obj.train_prototxt,'train');
            obj.action.forward          = @()caffe('forward');
            obj.action.forward_backward = @()caffe('forward_backward');
            obj.action.backward         = @(diffs)caffe('backward',diffs);
            obj.action.training_iter    = @(batch)caffe('training_iter',batch);
            
            obj.action.reset();
            
            obj.net_structure = net_struct;
        end
        
        %% SETTERS
        function set_use_gpu(obj,device_id)
            if device_id>=0
                obj.use_gpu = 1;
                %gpuDevice([]); %cleanup
                %            obj.action.reset();
                g = parallel.gpu.GPUDevice.getDevice(device_id+1);
                if ~g.DeviceSelected
                    gpuDevice(device_id+1); %select / handle_erroneous_id
                    caffe('set_device',device_id); %set to caffe
                end
                APP_LOG('info','GPU Device set to %s',g.Name);
            else
                obj.use_gpu = 0;
            end
        end
        
        function set_phase(obj,phase)
            obj.action.reset();
            phase=lower(phase);
            switch phase
                case 'train'
                    if isempty(obj.train_prototxt)
                        obj.train_prototxt = obj.net_structure.create_prototxt('train');
                    end
                    caffe('init',obj.train_prototxt,'train');
                    obj.current_phase='train';
                case 'validation'
                    if isempty(obj.validation_prototxt)                    
                        obj.validation_prototxt = obj.net_structure.create_prototxt('validation');                    
                    end
                    caffe('init',obj.validation_prototxt,'test');
                    obj.current_phase='validation';                    
                case 'test'
                    if isempty(obj.test_prototxt)
                        obj.test_prototxt       = obj.net_structure.create_prototxt('test');
                    end
                    caffe('init',obj.test_prototxt,'test');
                    obj.current_phase='test';
                otherwise
                    APP_LOG('last_error','Invalid phase. Expected train/validation/test, got %s',phase);
            end
            obj.set.params();
        end
        
        function set_labels(obj,arg_dataset)
            last_layer_bottom_blob = obj.net_structure.layers{end}.bottom_size;
            if length(arg_dataset)== last_layer_bottom_blob(3)
                for i=length(arg_dataset):-1:1
                    obj.labels(i).name       = arg_dataset(i).labels.name;
                    obj.labels(i).dataset_ID = arg_dataset(i).labels.dataset_ID;
                end
            else
                APP_LOG('error','Last Layer''s input size isn''t coherent with labels size');
                APP_LOG('last_error','[labels: %d - last layer''s bottom: [%d %d %d]',length(arg_dataset),last_layer_bottom_blob(1),last_layer_bottom_blob(2),last_layer_bottom_blob(3));
            end
        end
               
        function reset_object_input(obj,phase,object_input_size)
            switch(phase)
                case 'train'
                    obj.net_structure.set_batch_size(phase,object_input_size);
                    obj.net_structure.validate_structure();
                    obj.train_prototxt = obj.net_structure.create_prototxt(phase);                    
                case 'validation'
                    obj.net_structure.set_batch_size(phase,object_input_size);
                    obj.net_structure.validate_structure();
                    obj.validation_prototxt = obj.net_structure.create_prototxt(phase);                    
                case 'test'
                    obj.net_structure.set_batch_size(phase,object_input_size);
                    obj.net_structure.validate_structure();
                    obj.test_prototxt = obj.net_structure.create_prototxt(phase);
                otherwise
                    APP_LOG('last_error','Invalid phase "%s". Use train/validation/test instead',phase);
            end
            if strcmp(phase,obj.current_phase)
                obj.set_phase(phase);
            end
        end
        %% FUNCTIONS
        function init(obj,phase)          
            if obj.get.is_initialized() == 1
                obj.action.reset();
            end            
            % Validate network structure
            obj.net_structure.validate_structure();
            obj.set_phase(phase);
            
            % set to use GPU or CPU
            if obj.use_gpu
                caffe('set_mode_gpu');
            else
                caffe('set_mode_cpu');
            end
            
            obj.set.params();
        end
	end
end
