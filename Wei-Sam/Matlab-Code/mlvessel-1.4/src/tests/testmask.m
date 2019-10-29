function testmask(config)
% testmask(config)
%
% Outputs masks for all the images indicated in "config".
%
% See also: driveconfig, stareconfig.

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

% The feature manipulation module.
addpath([installdir filesep 'src' filesep 'ftrs']);
% The html module.
addpath([installdir filesep 'src' filesep 'html']);

% Directory with images.
addpath(config.imagedir);

% Creates features for all images.
imagenames = config.testnames;

% Creating directory for results.
resultdir = config.resultdir;
testname = ['mask_' config.testname];
ignore = mkdir(resultdir, testname);
resultdir = [resultdir testname filesep];

for i = 1:size(imagenames, 2)
  imagename = imagenames{i};
  name = imagename.original;
  
  shortname = name(1:(end-4));
  outputdir = [resultdir shortname filesep];
  ignore = mkdir(resultdir, shortname);

  mask_page(outputdir, name, config.imagetype);
end

function mask_page(outputdir, name, imagetype)

text = ['Retina Image Processing for ' name '.'];
fp = htopen([outputdir 'index.html'], text);
htwtext(fp, ['<h2> ' text ' </h2>']); 

img = imread(name);
shortname = name(1:(end-4));

htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Original image';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, img, [shortname '.jpg'], 3);

switch(imagetype)
 case 'angiogram'
  mask = createretinamaskangiogram(img);
 case 'colored'
  mask = createretinamaskcolored(img);
 case 'redfree'
  mask = createretinamaskredfree(img);
 otherwise
  disp(['Unknown image type ' config.imagetype]);
  return;
end

htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Mask image'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, mask, [shortname '-mask.png'], 3);

masked = img;
if ( size(img, 3) == 3 )
  red = img(:,:,1);
  green = img(:,:,2);
  blue = img(:,:,3);
  
  red(~mask) = 0;
  green(~mask) = 0;
  blue(~mask) = 0;
  
  masked(:,:,1) = red;
  masked(:,:,2) = green;
  masked(:,:,3) = blue;
else
  if strcmp(imagetype, 'redfree')
    masked = uint8(255 - double(masked));
  end
  masked(~mask) = 0;  
end

htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Masked image'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, masked, [shortname '-masked.jpg'], 3);

htclose(fp);