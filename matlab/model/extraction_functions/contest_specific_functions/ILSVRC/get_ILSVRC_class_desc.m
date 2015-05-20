function map = get_ILSVRC_class_desc(contest_file)
%GET_ILSVRC_CLASS_DESC Create a map for WNID->name

%Input Arguments :
%                  contest_filepath

%Output Arguments: map with 1-1 relation classID -> name of object in human
%                  understandable form.

try
    APP_LOG('info',4,'Loading ILSVRC contest metadata file');
    contest_meta = load(contest_file);
    struct_field = char(fieldnames(contest_meta));
    synsets = contest_meta.(struct_field);
    APP_LOG('info',4,'Loading complete');
catch err
    APP_LOG('last_error',0,'%s',err.message);
end
%create WNID map
APP_LOG('info',4,'Create Word-Net ID MAP');

map = containers.Map('KeyType','char','ValueType','char');
for i=1:length(synsets)
    map(synsets(i).WNID)=synsets(i).name;
end

end

