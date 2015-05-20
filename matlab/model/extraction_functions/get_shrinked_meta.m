function output = get_shrinked_meta(data,rmap,classID )
%GET_SHRINKED_meta Returns a shrinked metadata table for a specific dataset
%of a specific classID. This function is used to reduce the memory size
%that workers need in case of very large contests like... ILSVRC.

%Create shrinked meta table for this class of this set 
%(less ram usage for parfor, because... MPI)

%AUTHOR: PROVOS ALEXIS
%DATE:   18/5/2015
%FOR:    vision team - AUTH

indices= rmap(classID);
output = data(unique(indices(:)));
for i=1:length(output)
    %clear any classes on this entry that do not match with the specified
    %classID
    output(i).objs(~ismember({output(i).objs(:).name},{classID})) = [];
end
end

