classdef KLSH < handle
    %KLSH Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        weights;
        connections;
        clusters;
        x;
        L;
        fval;
        
    end
    
    methods
        function obj = KLSH()
            tmp = load('garesult15.mat'); % loads x,L,fval
            obj.x = tmp.x;
            obj.L = tmp.L;
            obj.fval = tmp.fval;
            
            tmp = load('clusters.mat');
            obj.clusters = tmp.clusters;
            
            tmp = load('connections.mat');
            obj.connections = tmp.connections;
            
            tmp = load('weights.mat');
            obj.weights = tmp.weights;
        end
        
        function out = get_prediction(obj,img)
            queryFeatures = singleExtraction( img,obj.x); %create query Features            
            testIDs = createTestIDs(queryFeatures,obj.L); %Run experiment for query
            scores = getHist(obj.clusters, testIDs)'; %classHist=scores;
            out = sub2class( scores ,obj.connections,obj.weights);
        end
    end
    
end

