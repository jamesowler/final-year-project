function testpad(config)
% testmask(config)
%
% Outputs results from applying the pad to all the images indicated in
% "config".
%
% See also: testmask, driveconfig, stareconfig.

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
testname = ['pad_' config.testname];
ignore = mkdir(resultdir, testname);
resultdir = [resultdir testname filesep];

for i = 1:size(imagenames, 2)
  imagename = imagenames{i};
  name = imagename.original;
  
  shortname = name(1:(end-4));
  outputdir = [resultdir shortname filesep];
  nada = mkdir(resultdir, shortname);

  pad_page(outputdir, name, config);
end

function pad_page(outputdir, name, config)

text = ['Retina Image Processing for ' name '.'];
fp = htopen([outputdir 'index.html'], text);
htwtext(fp, ['<h2> ' text ' </h2>']); 

img = imread(name);
shortname = name(1:(end-4));

htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Original image';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, img, [shortname '.jpg'], 3);

% Finds the aperture mask.
switch(config.imagetype)
 case 'angiogram'
  mask = createretinamaskangiogram(img);
 case 'colored'
  mask = createretinamaskcolored(img);
 case 'redfree'
  mask = createretinamaskredfree(img);
end

% Erosion parameters.
erosionsize = config.paderosionsize;
iterations = 80;

htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Mask image'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, mask, [shortname '-mask.png'], 3);

if (size(img, 3) == 3)
  img = double(img(:,:,2)) / 255;
  img = 1 - img;
else
  img = double(img) / 255;
end

htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Masked image'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, img .* double(mask), [shortname ...
                    '-masked.jpg'], 3);

% Also saves slightly smaller mask, which is actally used for padding.
smallmask = mask;

[nrows, ncols] = size(smallmask);
smallmask(1,:) =     zeros(1, ncols);
smallmask(nrows,:) = zeros(1, ncols);
smallmask(:,1) =     zeros(nrows, 1);
smallmask(:,ncols) = zeros(nrows, 1);

smallmask = imerode(smallmask, strel('disk', erosionsize, 0));

htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Masked image with smaller mask'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, img .* double(smallmask), ...
              [shortname '-small-masked.jpg'], 3);

% Enlarges image size.
[sizey, sizex] = size(img);

bigimg = zeros(sizey + 100, sizex + 100);
bigimg(51:(50+sizey), 51:(50+sizex)) = img;

bigmask = logical(zeros(sizey + 100, sizex + 100));
bigmask(51:(50+sizey), (51:50+sizex)) = mask;

% Saves enlarged image.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Big image with mask'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, bigimg .* double(bigmask), ...
              [shortname '-big.png'], 3);

% Creates artificial extension of image.
padded = fakepad(bigimg, bigmask, erosionsize, iterations);

% Saves padded enlarged image.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle = ['Padded image'];
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, padded, [shortname '-padded.png'], 3);
  
htclose(fp);