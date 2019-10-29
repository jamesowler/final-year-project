function createfeaturescolored(img, scales, paderosionsize, ...
                               outputfilename)
% createfeaturescolored(img, scales, paderosionsize,
% outputfilename)
%
% Recieves a colored non-mydriatic image, generates and writes each
% pixel's features on file named "outputfilename". "scales" is a
% vector indicating what scales of the Morlet wavelet are used for
% features. "paderosionsize" indicates the size of the erosion
% performed before extending the image with our padding. Features are
% saved in form of a three-dimensional matrix indexad by (x, y, c),
% where (x, y) is the pixel's position in the image and (c) is a
% feature.

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
mask = createretinamaskcolored(img);

% Generates the features.
[description, features] = generatefeaturescolored(img, mask, ...
                                                  scales, paderosionsize);

% Saves in "outputfilename" in matlab format.
save(outputfilename, 'description', 'features', 'mask', '-MAT');