%BUILD MATCAFFE.CU FOR LINUX

if isunix
    myArch = computer('arch');
    pathToOpts = fullfile(matlabroot,'toolbox','distcomp','gpu','extern','src','mex',myArch,'gcc',['mex_CUDA_' myArch '.xml']);
    copyfile(pathToOpts,'.','f')
    system('make matcaffe -j 4')
    cd matlab
    startup;
end

if ispc
    fprintf('Needs a windows pc to add support...');
end