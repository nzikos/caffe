sets  = ({'training','validation'});
meta  = fullfile('/','media','alexis','SSD','XML');  %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
imdb  = fullfile('/','media','alexis','SSD','IMDB'); %Must contain 2 subfolders named 'training' and 'validation' or whatever names were given to sets
cache = fullfile('/','media','alexis','SSD','CACHE');
dims  = [224 224];
contest.name = 'ILSVRC';


model = extraction_model();
model.set_sets(sets);
model.set_paths(meta,imdb,cache);
model.paths.print_paths();
model.set_contest(contest);

try
    model.load_metadata();
catch err
    APP_LOG('warning',0,'%s',err.message);
    model.build_metadata();
    model.check_metadata();
    model.save_metadata();
end

model.print_metadata();

model.set_objects(dims);

try
	model.load_objects();
catch err
    APP_LOG('warning',0,'%s',err.message);
    model.build_objects();
    model.save_objects();
end