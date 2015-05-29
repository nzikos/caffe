function [set,get,action] = get_caffe_mex_handlers()
%CAFFE_MEX_HANDLERS USED TO PASS THE HANDLERS OF THE MEX FILE TO NETWORK
%% Description
%   This function is returning all the mex API commands to caffe from
%   matlab to the network structure so that they can be used during
%   training validation

%% CAFFE SETTERS HANDLERS
set.phase.train         = @()caffe('set_phase','train');
set.phase.test          = @()caffe('set_phase','test');
set.weights             = @(w)caffe('set_weights',w);
set.input               = @(i)caffe('upload_input',i);
set.device              = @(id)caffe('set_device',id);
%% CAFFE GETTERS HANDLERS
get.weights             = @()caffe('get_weights');
get.output              = @()caffe('download_output');
get.grads               = @()caffe('get_grads');
get.is_initialized      = @()caffe('is_initialized');
%% CAFFE ACTIONS HANDLERS
action.reset            = @()caffe('reset');
action.forward          = @()caffe('forward');
action.forward_backward = @()caffe('forward_backward');
action.backward         = @(diffs)caffe('backward',diffs);
action.training_iter    = @(batch)caffe('training_iter',batch);
end

