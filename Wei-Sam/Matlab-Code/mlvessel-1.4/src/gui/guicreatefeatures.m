function guicreatefeatures(img, imagetype, featurestype, outputfilename)
% guicreatefeatures(img, imagetype, featurestype, outputfilename)
%
% Recieves an image, its type and a cell of structs defining the
% features to be created (featurestype). The features are generated
% and written on file named "outputfilename". Features are saved in
% form of a three-dimensional matrix indexad by (x, y, c), where 
% (x, y) is the pixel's position in the image and (c) is a feature.

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

% Creates the aperture mask, with the image's region of interest.
[newfeatures, mask] = guigeneratefeatures(img, imagetype, featurestype);

% Changes format of features.
% Creates 3D feature matrix with data and also the description matrix
description = [];
features = [];
for i = 1:length(newfeatures)
    features = cat(3, newfeatures{i}.data, features);
    description = strvcat(newfeatures{i}.name, description);
end

% Saves in "outputfilename" in matlab format.
save(outputfilename, 'description', 'features', 'mask', '-MAT');