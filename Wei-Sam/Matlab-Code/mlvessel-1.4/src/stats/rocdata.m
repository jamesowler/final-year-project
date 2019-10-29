function [fp, tp, n, p] = rocdata(control, test, aperture, span)
% [fp, tp, n, p] = rocdata(control, test, aperture, span)
%
% Returns the ROC data for image "test", thresholded at values in
% "span". "control" contains the ground truth and "aperture" is the
% region of interest. The data returned is:
%
% fp - vector of false positive counts indexed by values in "span"
% tp - vector of true positive counts indexed by values in "span"
%  p - total value of positives
%  n - total value of negatives
%
% See also: rocarea.

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

negativeimage = ~control & aperture;
n = sum(negativeimage(:));
 
positiveimage = control & aperture;
p = sum(positiveimage(:));

tp = histc(test(positiveimage(:)), span);
tp = cumsum(tp(end:-1:1)');
tp = tp(end:-1:1);

fp = histc(test(negativeimage(:)), span);
fp = cumsum(fp(end:-1:1)');
fp = fp(end:-1:1);