function vector = vectorize_objects_fpaths(fpaths)
%VECTORIZE_OBJECTS_FPATHS This function is used to vectorize objects
%filepaths from a cell array of cells organized by class to a 1-D cell 
%array of filepaths.
%The 'vector' will be used as a pool by the network to
%feed the training/validation processes with objects.

vector = {};
index=1;

%Make sure that classes with minimum number of objects get first into the vector
for i=1:length(fpaths)
    amount(i)=length(fpaths(i).paths);
end
[~,sorted_indices]=sort(amount,'ascend');

%Create vector
for oo=1:length(sorted_indices)
    i=sorted_indices(oo);
    for j=1:length(fpaths(i).paths)
        vector{index,1} = fpaths(i).paths{j};
        index=index+1;
    end
end


end

