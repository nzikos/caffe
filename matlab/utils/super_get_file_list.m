function list = super_get_file_list( directory,extension,path)
%SUPER_GET_FILE_LIST Returns all filepaths of files with specified
%extension searching recursively on all sub directories that might exist.
%
%input:
%   directory: root directory in which to search 
%              (OS dependent, so use fullfile to create a directory)
%   extension: .extension of file (dont miss the dot)
%
%output (array of structures)
%   filedir  : absolute directory of found file
%   path     : relative paths to the folder this file is found
%
%AUTHOR: PROVOS ALEXIS

if nargin<3
    path='';
end
list=[];
%% GET SUBDIRECTORIES
sub_folders=get_sub_dirs(directory);

if isempty(sub_folders)
    %% IF NO SUBFOLDERS -> GET FILES (IF ANY)
    filedir_list = get_file_list(directory,extension);
    for i=1:length(filedir_list)
        list(i).filedir = char(filedir_list{i});
        list(i).path    = path;
    end
else
    %% ELSE MOVE INTO EACH SUBFOLDER (IN PARALLEL-yay)
    parfor i=1:length(sub_folders)
        new_path=fullfile(path,sub_folders{i});
        sub_dir=fullfile(directory,sub_folders{i});
        list = [list super_get_file_list(sub_dir,extension,new_path)];
    end
    %% MERGE PARALLEL WORK DONE - 50% slower
%    list=[];
%    for i=1:length(sub_folders)
%        list=[list p_list{i}];
%    end
end
end

