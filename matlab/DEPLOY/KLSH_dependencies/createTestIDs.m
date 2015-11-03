function clusters = createTestIDs(features, L)

for i=1:L.noOfExperiments
    clusters.exp(i).id = [];
    clusters.exp(i).population = [];
end

for i=1:length(features)
    tmpCluster = createClusterForSingleClass(L, features(i).F, i, length(features));
    clusters = mergeClusters(clusters, tmpCluster);
end
