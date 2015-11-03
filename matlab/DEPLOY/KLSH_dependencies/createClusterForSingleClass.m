function clusters = createClusterForSingleClass(L , F, classId, noOfClasses)

%%
%find the hash keys for each experiment
%for i=1:size(F,2)
%    id(:,i) = floor(L.A*F(:,i) + L.B);
%end
id = floor( L.A*F + repmat(L.B,1,size(F,2)));
%%
%for each experiment do clustering
tmpPopulation = zeros(size(F,2), noOfClasses);
tmpPopulation(:,classId) = 1;

for i=1:L.noOfExperiments
    %choose the hash table for each experiment
    clusters.exp(i).id = id( (i-1)*L.k+1:i*L.k, : )';
    clusters.exp(i).population = tmpPopulation;
end