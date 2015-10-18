%STARTUP.m Add paths to matlab search path
%This script is running by default when you start matlab on this directory
%It's purpose is to add all functions stored for this project into the
%search path of matlab environment.

%a=gpuArray(rand(1)); clear a;

%DIRECTORY OF MODEL related files
s_dirs =    genpath('model');
addpath(s_dirs);

%DIRECTORY OF NETWORK FUNCTIONS
s_dirs =    genpath('network');
addpath(s_dirs);

%DIRECTORY OF DEPLOYED NETWORK
s_dirs =    genpath('DEPLOY');
addpath(s_dirs);

%DIRECTORY OF CAFFE
s_dirs =    genpath('caffe');
addpath(s_dirs);

%DIRECTORY OF PRETRAINED LAYER DATA-FILLERS
%s_dirs =    genpath('pretrained_functions');
%addpath(s_dirs);

%DIRECTORY OF UTILS
s_dirs =    genpath('utils');
addpath(s_dirs);

clear all;
