% This code was originally written and distributed as part of the
% PASCAL VOC challenge. But now...
% This code was modified.

%   This is a function from the contest_specific_functions set.
%   This function's action is to take a list of metadata filepaths and
%   their relative paths and create an array of structs as described under
%   build_meta.m
%   

function out = ILSVRC_readmeta(list,imdb_dir,imdb_ext)

parfor i=1:length(list)
    filepath        = list(i).filedir;
    rel_path        = list(i).path;

    [~,filename,~]  = fileparts(list(i).filedir);
    imdb_cor        = fullfile(imdb_dir,rel_path,[filename imdb_ext]);

    f=fopen(filepath,'r');
    xml=fread(f,'*char')';
    fclose(f);
    temp=VOCxml2struct(xml);
    if ~isfield(temp.annotation,'object')
        temp.annotation.object = [];
    end

    out(i) = data2struct(temp.annotation,imdb_cor);
end

end

function out = data2struct(in,imdb_cor)
    
    out.imdb_cor    =imdb_cor;
    
    out.size        =[str2num(in.size.height) str2num(in.size.width)];
    
    if isempty(in.object)
        out.objs = [];
        return;
    else
        for i=1:length(in.object)
            obj.name         = in.object(i).name;
            obj.bndbox.xmin  = str2num(in.object(i).bndbox.xmin);
            obj.bndbox.xmax  = str2num(in.object(i).bndbox.xmax);
            obj.bndbox.ymin  = str2num(in.object(i).bndbox.ymin);
            obj.bndbox.ymax  = str2num(in.object(i).bndbox.ymax);                
            out.objs(i) = obj;
        end
    end
end

%--------------------------------------------------------------------------
% This code was originally written and distributed as part of the
% PASCAL VOC challenge
function res = VOCxml2struct(xml)

xml(xml==9|xml==10|xml==13)=[];

[res,~]=parse(xml,1,[]);
end
%--------------------------------------------------------------------------
%Subfunction
function [res,ind]=parse(xml,ind,parent)

res=[];
if ~isempty(parent)&&xml(ind)~='<'
    i=findchar(xml,ind,'<');
    res=trim(xml(ind:i-1));
    [tag,ind]=gettag(xml,i);
    if ~strcmp(tag,['/' parent])
        error('<%s> closed with <%s>',parent,tag);
    end
else
    while ind<=length(xml)
        [tag,ind]=gettag(xml,ind);
        if strcmp(tag,['/' parent])
            return
        else
            [sub,ind]=parse(xml,ind,tag);            
            if isstruct(sub)
                if isfield(res,tag)
                    n=length(res.(tag));
                    fn=fieldnames(sub);
                    for f=1:length(fn)
                        res.(tag)(n+1).(fn{f})=sub.(fn{f});
                    end
                else
                    res.(tag)=sub;
                end
            else
                if isfield(res,tag)
                    if ~iscell(res.(tag))
                        res.(tag)={res.(tag)};
                    end
                    res.(tag){end+1}=sub;
                else
                    res.(tag)=sub;
                end
            end
        end
    end
end
end
%--------------------------------------------------------------------------
%Subfunction
function i = findchar(str,ind,chr)

i=[];
while ind<=length(str)
    if str(ind)==chr
        i=ind;
        break
    else
        ind=ind+1;
    end
end
end
%--------------------------------------------------------------------------
%Subfunction
function [tag,ind]=gettag(xml,ind)

if ind>length(xml)
    tag=[];
elseif xml(ind)=='<'
    i=findchar(xml,ind,'>');
    if isempty(i)
        error('incomplete tag');
    end
    tag=xml(ind+1:i-1);
    ind=i+1;
else
    error('expected tag');
end 
end
%--------------------------------------------------------------------------
%Subfunction
function s = trim(s)

for i=1:numel(s)
    if ~isspace(s(i))
        s=s(i:end);
        break
    end
end
for i=numel(s):-1:1
    if ~isspace(s(i))
        s=s(1:i);
        break
    end
end
end