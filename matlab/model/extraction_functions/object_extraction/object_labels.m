function labels = object_labels(class_names_map,classes,j)
%OBJECT_CLASS_LABELS Creates the struct with class info that will be
%
%%  Appended into each object
%   labels.name       = The name of the class from contest map
%   labels.dataset_ID = The uid that describes this class under metadata
%                       (WNID in case of ILSVRC)
%   labels.vector     = The vector that will be supplied to caffe
%   labels.uid        = The class uid in integer format (for caffe too)
%
%%  AUTHOR: PROVOS ALEXIS
%   DATE:   20/5/2015
%   FOR:    VISION TEAM - AUTH

    labels.name       = class_names_map(classes{j});
    labels.dataset_ID = classes{j};
    labels.uid        = j-1; %caffe is 0-based
    
end

