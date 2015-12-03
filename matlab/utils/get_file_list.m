function [list] = get_file_list(datadir,filetypes)
%get_file_list export a list of files from a specific dir
%
%   This function is accessing the specified directory
%   and returns a list of all files included
%
%   This is a sub-function of super_get_file_list which implements
%   reccursion.

dirStr          = dir(datadir);
list            = [];
fileIdx         = 1;
for ff = 1:length(dirStr)
    f_name = dirStr(ff).name;
    if(~strcmp(f_name,'.') && ~strcmp(f_name,'..') && ~dirStr(ff).isdir)
        [~, ~, ext] = fileparts(f_name);
        if strcmp(ext,filetypes)
            list{fileIdx,1} = fullfile(datadir,f_name);
            fileIdx = fileIdx + 1;
        end
    end
end
end
