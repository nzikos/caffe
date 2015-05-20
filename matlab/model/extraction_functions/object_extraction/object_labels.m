function labels = object_labels(class_names_map,classes,j)
%OBJECT_CLASS_LABELS Creates the struct with class info that will be
%appended into each object
%   labels.name      = The name of the class from contest map
%   labels.contestID = The uid that describes this class under metadata
%                     (WNID in case of ILSVRC)
%   labels.vector    = The vector that will be supplied to caffe
%   labels.uid       = The class uid in integer format (for caffe too)

    labels.name      = class_names_map(classes{j});
    labels.contestID = classes{j};
    labels.vector    = zeros(length(classes),1);
    labels.vector(j) = 1;
    labels.uid       = j-1; %caffe is 0-based
end

