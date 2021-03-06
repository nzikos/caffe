classdef DATASETS < handle
    %DATASETS contains the extensions of metadata and image files as also
    %the function handlers who build the metadata and extract the objects.
    
    %Currently supported DATASETS: ILSVRC_DET
    %properties:
    %   meta_extension: The metadata files extension
    %   im_extension:   The image files extension
    %   name:           The dataset name;
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
    %   class_names_map:Is a containers.map which correlates dataset
    %                   classes to human understandable text. E.g. WordNet
    %                   classes.
    %
    %%   AUTHOR: PROVOS ALEXIS
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
        function obj = DATASETS(in)
            if ~isfield(in,'name')
                APP_LOG('last_error','You need to pass input.name with name being the dataset name');
            end
            switch(in.name)
                case 'ILSVRC_DET'
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
                    APP_LOG('info','Dataset set to ILSVRC_DET');
                case 'User defined'
                    obj.name = in.name;
                    obj.meta_extension = in.meta_extension;
                    obj.im_extension   = in.im_extension;
                    obj.read_meta      = in.read_meta;
                    obj.extract_objects= in.extract_objects;
                    obj.class_names_map= in.class_names_map;
                otherwise
                    APP_LOG('last_error','Dataset %s Not supported. Use "User defined" and supply the appropriate functions',in.name);
            end
        end
    end
    
end

