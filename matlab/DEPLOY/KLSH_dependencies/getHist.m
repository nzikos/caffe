function classHist = getHist(clusters, testIDs)

classHist = [];
for i=1:length(clusters.exp)
    classHist(:,:,i) = getHistSingleExp(clusters.exp(i).id, ...
                       clusters.exp(i).population, testIDs.exp(i).id);    
end
classHist = sum(classHist,3);
sums=sum(classHist,2);
sums(sums==0)=1;
classHist=classHist./repmat(sums,1,size(classHist,2));


function hist = getHistSingleExp(bucketID, population, qIDs)

[fIndex, idx] = ismember(qIDs, bucketID, 'rows');
hist = zeros(size(qIDs,1),size(population, 2));
hist(fIndex, :) = population(idx(fIndex), :);