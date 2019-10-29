function [skelimage, final] = skeleton(seg, flag, img)
% [skelimage, final] = skeleton(seg, flag, img)
%
% [skelimage, final] = skeleton(seg, flag)
%
% Receives a segmentation "seg", "flag" and, optionally the original
% image "img". Pre-processes "seg", extracts the skeleton and returns
% it in "skelimage". If "img" is provided, the return parameter "final" is
% produced, containing the skeleton overlayed on the original
% image. "flag" should be 1 for the manual removal of the optic disk
% region and 0 otherwise.
%
% See also: polymasknewbg

%
% Copyright (C) 2002 Emerson Luiz Navarro Tozette
%               2006  João Vitor Baldini Soares
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

if ( (size(seg, 3) == 1) & ~islogical(seg) )
  seg = seg > 126;
end

% Pre-processing.
% Tries to adjust parameters to image size with "sizefactor".
sizefactor = 1 + round(size(seg, 2) / 500);
% Removes small noisy regions.
seg = bwareaopen(seg, 4 * sizefactor);
% Dilates to connect close regions.
seg = imdilate(seg, strel('disk', sizefactor, 0));
% Removes larger isolated regions.
seg = bwareaopen(seg, 40 * sizefactor);
% Fills in small holes.
seg = bwareaclose(seg, 8 * sizefactor);

% These borders are necessary to run the skeletonization algorithm.
[nlins,ncols] = size(seg);
seg(1,:) =     ones(1, ncols);
seg(nlins,:) = ones(1, ncols);
seg(:,1) =     ones(nlins, 1);
seg(:,ncols) = ones(nlins, 1);

seg(2,:) =         ones(1, ncols);
seg(nlins - 1,:) = ones(1, ncols);
seg(:,2) =         ones(nlins, 1);
seg(:,ncols - 1) = ones(nlins, 1);

% Now, the skeletonization.
% label is an image of 4-connected labels.
[label, n] = bwlabel(seg, 4);

[CONT,DIL,SKEL] = skel(label, 100); %% space-scale skeletonization.
skelimage = (SKEL > (3 * sizefactor)) & seg;
% Makes skeleton as thin as possible.
skelimage = bwmorph(skelimage, 'skel', Inf);

if ( nargin > 2 ) % "img" is present.
  if ( size(img, 3) == 1 ) % transforms gray image in colored.
    img = repmat(img, [1 1 3]);
    
    skelcolor = [0, 0, 255];
  else
    skelcolor = [255, 255, 255];
  end
end

% Interacts with user for optic disk removal if necessary.
if ( flag == 1 )
  
  if ( nargin > 2 ) % "img" is present.
    for i = 1:3
      canal = img(:,:,i);
      canal(skelimage) = skelcolor(i);
      tempimg(:,:, i) = canal;
    end
  else 
    tempimg = skelimage;
  end
  
  diskmask = polymasknewbg(tempimg);
  
  skelimage = skelimage & diskmask;
end

% Builds the "final" image if "img" input parameter is present.
if ( nargin > 2 )
  skeldil = imdilate(skelimage, strel('disk', 2, 0));
  
  for i = 1:3
    canal = img(:,:,i);
    canal(skeldil) = skelcolor(i);
    final(:,:, i) = canal;
  end
end


function bw2 = bwareaclose(bw1, n)

bw2 = ~bwareaopen(~bw1, n);
