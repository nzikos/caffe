classdef CONTEST
    %CONTEST contains the extensions of metadata and image files as also
    %the function handlers who build the metadata and extract the objects.
    
    %Currently supported contests: ILSVRC
    %properties:
    %   meta_extension: The metadata files extension
    %   im_extension:   The image files extension
    %   name:           The contest name;
    %   read_meta:      The function handler who reads the metadata. As
    %                   input takes a list of all 'meta_extension'
    %                   filepaths found under root dir 
    %                   'paths.metadata.(set(x))'. It's also using as input
    %                   imdb_dir and imdb_ext to create the
    %                   imdb_cor(respondence) attribute for the
    %                   object_extraction process.
    %
    %   extract_objects:The function handler which extracts the objects. As
    %                   input it takes the metadata array of structs 
    %                   produced from read_meta, the dims of the extracted
    %                   object, the labels that will be appended to this
    %                   object, and the directory where this object will be
    %                   saved.
    %   class_names_map:Is a containers.map which correlates contest
    %                   classes to human understandable text. E.g. WordNet
    %                   classes.
    
    %   AUTHOR: PROVOS ALEXIS
    %   DATE:   19/5/2015
    %   FOR:    vision team - AUTH
    properties
        meta_extension;
        im_extension;
        name;
        read_meta;
        extract_objects;
        class_names_map;
    end
    
    methods
        function obj = CONTEST(in)
            if ~isfield(in,'name')
                APP_LOG('error_lat',0,'You need to pass input.name with name being the contest name');
            end
            switch(in.name)
                case 'ILSVRC'
                    obj.meta_extension='.xml';
                    obj.im_extension='.JPEG';
                    obj.name = in.name;
                    obj.read_meta = @(list,...
                                      imdb_dir,...
                                      imdb_ext...
                                    )ILSVRC_readmeta(list,...
                                                     imdb_dir,...
                                                     imdb_ext);
                                                 
                    obj.extract_objects = @(this_meta...
                                           ,dims...
                                           ,labels...
                                           ,appendTo...
                                           )ILSVRC_extract(this_meta...
                                                          ,dims...
                                                          ,labels...
                                                          ,appendTo...
                                                          );
                    contest_file = fullfile(pwd,'model','extraction_functions',...
                                        'contest_specific_functions',...
                                        'ILSVRC',...
                                        'meta_det.mat');
                                    
                    obj.class_names_map = get_ILSVRC_class_desc(contest_file);
                    APP_LOG('info',0,'Contest set to ILSVRC');
                case 'User defined'
                    obj.name = in.name;
                    obj.meta_extension = in.meta_extension;
                    obj.im_extension   = in.im_extension;
                    obj.read_meta      = in.read_meta;
                    obj.extract_objects= in.extract_objects;
                    obj.class_names_map= in.class_names_map;
                otherwise
                    APP_LOG('error_last',0,'Contest %s Not supported. Use "User defined" and supply the appropriate functions',in.name);
            end
        end
    end
    
end

