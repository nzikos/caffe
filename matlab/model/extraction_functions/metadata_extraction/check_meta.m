function check_meta(data,rmap,set)
%CHECK_METADATA Validates the classes according to rules set in
%here.
%   Rules:
%       1. Number of classes between sets must be the same.
%       2. one-to-one correspondence rule for classes of sets.

APP_LOG('info',0,'Checking metadata robustness between sets');

%% INITIALIZE
set_size=numel(set);
if set_size==1
    APP_LOG('error_last',0,'You need at least two sets!');
end

for i=1:set_size
    classes.(set{i}) = (rmap.(set{i}).keys)';
end

%% TEST 1
    err=0;
    for i=2:set_size
        if numel(classes.(set{1}))~=numel(classes.(set{i}))
            err=i;
            break;
        end
    end
    if err
        APP_LOG('last_error',0,'Number of %s set classes (%d) are unequal to number of %s set classes (%d)',set{1},numel(classes.(set{1})),set{err},numel(classes.(set{err})));
    else
        APP_LOG('info',1,'1st test passed');
    end    
%% TEST 2
    err=false;
    for i=1:length(classes.(set{1}))
        for j=2:set_size
            if ~rmap.(set{1}).isKey(classes.(set{j}){i})
                APP_LOG('error',0,'Class with %s unique id from %s set has no %s set correspondence',classes.(set{j}){i},set{j},set{1});
                APP_LOG('error',0,'Files from %s set with no %s set correspondence are:',set{j},set{1});
                idxs=rmap.(set{j})(classes.(set{j}){i})';
                for k=1:length(idxs);
                    APP_LOG('error',0,'%25s',data.(set{j})(idxs(k)).imdb_cor);
                end
                err=true;
                break;
            end
            if ~rmap.(set{j}).isKey(classes.(set{1}){i})
                APP_LOG('error',0,'Class with %s unique id from %s set has no %s set correspondence',classes.(set{1}){i},set{1},set{j});
                APP_LOG('error',0,'Files from %s set with no %s set correspondence are:',set{1},set{j});
                idxs=rmap.(set{1})(classes.(set{1}){i})';
                for k=1:length(idxs);
                    APP_LOG('error',0,'%25s',data.(set{1})(idxs(k)).imdb_cor);
                end
                err=true;
                break;        
            end    
        end
    end
    if err
        APP_LOG('error_last',0,'Terminating');
    else
        APP_LOG('info',1,'2nd test passed');    
    end    
end