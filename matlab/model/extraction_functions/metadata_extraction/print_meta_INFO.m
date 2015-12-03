%PRINT_INFO Prints information about metadata
%
%%  AUTHOR: PROVOS ALEXIS
%   DATE:   19/5/2015
%   FOR:    vision team - AUTH
function print_meta_INFO(data,rmap,set,class_names_map)
classes = rmap.(set{1}).keys;

% PRINT HEADER
out = sprintf('%4s%22s[%9s]','N','Name','classID');
for i=1:length(set)
    out = [out sprintf('%10s',set{i})];
end
APP_LOG('header','%s',out);

% Init total_objs
for i=1:length(set)
    total_objs.(set{i})=0;
end

% PRINT info for each class
for j=1:length(classes)
    
    out = sprintf('%4s%22s[%9s]',num2str(j),class_names_map(classes{j}),classes{j});    

    for i=1:length(set)
        temp_meta = get_shrinked_meta(data.(set{i}),rmap.(set{i}),classes{j});
        objects=0;
        for k=1:length(temp_meta)
            for l=1:length(temp_meta(k).objs)
                objects=objects+1;
                total_objs.(set{i})=total_objs.(set{i})+1;
            end
        end
        out = [out sprintf('%10s',num2str(objects))];
    end
    APP_LOG('info','%s',out);
end 

for i=1:length(set)
	APP_LOG('header','%s SET',upper(set{i}));
    unclassified=0;
    set_size=length(data.(set{i}));
    for j=1:set_size
        if isempty(data.(set{i})(j).objs)
            unclassified=unclassified+1;
        end
    end
    if unclassified~=0
        APP_LOG('warning','%d/%d metadata files from %s set have no objects, will be ignored',unclassified,set_size,set{i});
    end
    APP_LOG('info','There are %d available objects in %s set',total_objs.(set{i}),set{i});
end

end

