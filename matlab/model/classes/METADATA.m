classdef METADATA < handle
    %METADATA contains the metadata array of structs extracted from
    %         paths.metadata directories.
    %Attributes:
    %           data: The metadata struct of arrays containing structs
    %                 extensively described under build_meta.m
    %           rmap: The Containers.map containing indices of data 
    %                 whose objs contain the specific classID. (Used for
    %                 better memory management on parfors).
    
    %Functions:
    %           build_metatada: Builds the metadata
    %                    check: Checks classes between sets for robustness
    %                    print: Prints a log of classes and objects on each
    %                           set.
    %            save_metadata: Saves the metadata under cache folder
    %            load_metadata: Loads the metadata from the specified cache
    %                           folder.
    
    properties
        data;
        rmap;
    end
    
    methods (Hidden = true)
        function obj = METADATA()
            obj.data=[];
            obj.rmap=[];
        end
        
        function build_metadata(obj,paths,dataset,sets)
            [obj.data,obj.rmap] = build_meta(paths,dataset,sets);
        end
        
        function check(obj,sets)
            check_meta(obj.data,obj.rmap,sets);
        end
        
        function print(obj,set,class_names_map)
            print_meta_INFO(obj.data,obj.rmap,set,class_names_map);
        end
        
        function save_metadata(obj,meta_file)
            APP_LOG('info','Save metadata...');
            data = obj.data;
            rmap = obj.rmap;
            save(meta_file,'data','rmap','-v6');
            APP_LOG('info','Metadata saved succesfully!');            
        end        
        
        function load_metadata(obj,meta_file)
            APP_LOG('header','Loading metadata from %s',meta_file);
            load(meta_file);
            obj.data = data;
            obj.rmap = rmap;
            APP_LOG('info','Metadata loaded succesfully!');
        end
    end
    
end

