function [features, mask] = guigeneratefeaturesredfree(img, featurestype)
% [features, mask] = guigeneratefeaturesredfree(img, featurestype)
%
% Recieves a gray-scale red-free retinal image, and a cell of structs
% defining the features to be created (featurestype). The features are
% generated and returned in a cell of structs "features". The function
% also return the mask defining the image's region of interest.
%

%
% Copyright (C) 2006  João Vitor Baldini Soares
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor,
% Boston, MA 02110-1301, USA.
%

if size(img, 3) == 3
    img = rgb2gray(img);
end

mask = createretinamaskredfree(img);

img = double(img) / 255;

% Inverting so vessels become brighter.
img = 1 - img;

% Makes the image larger before creating artificial extension, so the
% wavelet doesn't have border effects
[sizey, sizex] = size(img);

bigimg = zeros(sizey + 100, sizex + 100);
bigimg(51:(50+sizey), 51:(50+sizex)) = img;

bigmask = logical(zeros(sizey + 100, sizex + 100));
bigmask(51:(50+sizey), (51:50+sizex)) = mask;

% Creates artificial extension of image.
paderosionsize = round((sizex + sizey) / 250);
bigimg = fakepad(bigimg, bigmask, paderosionsize, 50);

% Below, creates the maximum wavelet response over angles and adds
% them as pixel features
fimg = fft2(bigimg);

features = featurestype;

for i=1:length(features)
    switch features{i}.type
        case 'Inverted gray channel'
            features{i}.data = bigimg(51:(50+sizey), (51:50+sizex));
            features{i}.name = 'Inverted gray channel';
            features{i}.shortname = 'gray';
        case 'Gabor processed inverted gray channel'
            k0x = 0;
            % Maximum transform over angles.
            trans = maxmorlet(fimg, features{i}.parameters.scale, ...
                features{i}.parameters.epsilon, [k0x features{i}.parameters.k0y], 10);
            trans = trans(51:(50+sizey), (51:50+sizex));
            
            % Adding to features
            features{i}.data = trans;
            features{i}.name = ['Gabor processed inverted gray channel: a = ' ...
                    num2str(features{i}.parameters.scale) ' eps = ' ...
                    num2str(features{i}.parameters.epsilon) ' k0y = ' ...
                    num2str(features{i}.parameters.k0y)];
            features{i}.shortname = ['gabor-a' num2str(features{i}.parameters.scale) ...
                    '-eps' num2str(features{i}.parameters.epsilon) ...
                    '-k0y' num2str(features{i}.parameters.k0y)];
    end
end