sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims  = [224 224];
contest.name = 'ILSVRC';


model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
%model.paths.print_paths();
model.set_contest(contest);

try
    model.load_metadata();
catch err
    APP_LOG('warning','%s',err.message);
    model.build_metadata();
    model.check_metadata();
    model.save_metadata();
end

%model.print_metadata();

%model.set_objects(dims);

try
	model.load_objects();
catch err
    APP_LOG('warning','%s',err.message);
    model.build_objects();
    model.save_objects();
end


for j=1:length(sets)
    for i=1:length(model.objects.data.(sets{j}))
        APP_LOG('header','%s objects for %s',sets{j},model.objects.data.(sets{j})(i).labels.name);
        if strcmp(model.objects.data.(sets{j})(i).labels.name,'sheep')
            for l=1:length(model.objects.data.(sets{j})(i).paths)
                load(model.objects.data.(sets{j})(i).paths{l})
                imshow(object.data);
                pause(0.5);
            end
        end
    end
end