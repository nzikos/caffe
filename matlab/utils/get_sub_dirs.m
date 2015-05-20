function sub_dirs = get_sub_dirs(directory)
%GET_SUB_DIRS Returns sub directories of input directory

contents = dir(directory);
sub_dirs = {contents([contents(:).isdir]).name}';
%perfom dummy cleanup
sub_dirs(ismember(sub_dirs,{'.','..'})) = [];

end

