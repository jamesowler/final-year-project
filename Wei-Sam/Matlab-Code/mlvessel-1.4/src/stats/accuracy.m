function [ac, true, total] = accuracy(control, test, aperture)
% [ac, true, total] = accuracy(control, test, aperture)
%
% Gives the accuracy for "test" using ground truth "control" for
% pixels in the region of interest "aperture".
%
% The return values are: 
%      ac - the accuracy
%    true - number of agreeing pixels in "control" ant "test" that
%           are in the "aperture".
%   total - total number of pixels in "aperture".
%
% So that we have ac = true / total.

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

trueimage = (control == test) & aperture;

true = sum(trueimage(:));
  
total = sum(aperture(:));

ac = true / total;
