%STARTUP.m Add paths to matlab search path
%This script is running by default when you start matlab on this directory
%It's purpose is to add all functions stored for this project into the
%search path of matlab environment.

%PLACE WHERE MODEL related files SHOULD BE FOUND
s_dirs =    genpath('model');
addpath(s_dirs);

%PLACE WHERE NETWORK FUNCTIONS SHOULD BE FOUND
s_dirs =    genpath('network');
addpath(s_dirs);

%PLACE WHERE CAFFE SHOULD BE FOUND
s_dirs =    genpath('caffe');
addpath(s_dirs);

%PLACE WHERE PRETRAINED LAYER DATA-FILLERS SHOULD BE FOUND
%s_dirs =    genpath('pretrained_functions');
%addpath(s_dirs);

%PLACE WHERE UTILS SHOULD BE FOUND
s_dirs =    genpath('utils');
addpath(s_dirs);

clear all;
