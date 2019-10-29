function [features, mask] = guigeneratefeatures(image, imagetype, featurestype)
% [features, mask] = guigeneratefeatures(image, imagetype, featurestype)
%
% Recieves an image, its type and a cell of structs defining the
% features to be created (featurestype). The features are generated
% and returned in a cell of structs "features". The function also
% return the mask defining the image's region of interest.

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

switch imagetype
case 'colored'
    [features, mask] = guigeneratefeaturescolored(image, featurestype);
case 'angiogram'
    [features, mask] = guigeneratefeaturesangiogram(image, featurestype);
case 'redfree'
    [features, mask] = guigeneratefeaturesredfree(image, featurestype);
end