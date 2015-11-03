function [ L,A,B ] = rgb2lab( image )
%Converts an rgb image to lab image
%L being the luminance

colorTransform = makecform('srgb2lab');
lab = applycform(image, colorTransform);
L = lab(:, :, 1);  % Extract the L image.
A = lab(:, :, 2);  % Extract the A image.
B= lab(:, :, 3);  % Extract the B image.
end

