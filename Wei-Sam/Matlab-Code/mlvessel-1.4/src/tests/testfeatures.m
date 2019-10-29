function testfeatures(config)
% testfeatures(config)
%
% Outputs features for all the images indicated in "config".
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
testname = ['features_' config.testname];
ignore = mkdir(resultdir, testname);
resultdir = [resultdir testname filesep];

for j = 1:size(imagenames, 2)
  imagename = imagenames{j};
  name = imagename.original;
  
  shortname = name(1:(end-4));
  outputdir = [resultdir shortname filesep];
  ignore = mkdir(resultdir, shortname);

  features_page(outputdir, name, config);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creates features and saves them in html format
%
function features_page(outputdir, name, config)
img = imread(name);
shortname = name(1:(end-4));

% Creates features
switch(config.imagetype)
 case 'angiogram'
  mask = createretinamaskangiogram(img);
  [description, features] = generatefeaturesangiogram(img, mask, ...
                                                  config.morletscales, ...
                                                  config.paderosionsize);
 case 'colored'
  mask = createretinamaskcolored(img);
  [description, features] = generatefeaturescolored(img, mask, ...
                                                    config.morletscales, ...
                                                    config.paderosionsize);
 case 'redfree'
  mask = createretinamaskredfree(img);
  [description, features] = generatefeaturesredfree(img, mask, ...
                                                    config.morletscales, ...
                                                    config.paderosionsize);
 otherwise
  disp(['Unknown image type ' config.imagetype]);
  return;
end

text = ['Retina Image Processing for ' name '.'];
fp = htopen([outputdir 'index.html'], text);
htwtext(fp, ['<h2> ' text ' </h2>']); 

htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Original image';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, img, [shortname '.jpg'], 4);

% Saves each feature created.
for i = 1:size(description, 1)
  feature = features(:,:,i);
  feature = feature .* double(mask);
  feature = feature ./ max(feature(:));
  
  htwpar(fp); htwhr(fp); htwpar(fp);
  httitle = description(i, :);
  htwtext(fp, ['<h2>' httitle '</h2>']);
  htwimage(fp, outputdir, feature, ...
           [shortname '-feature-' num2str(i) '.png'], 4);
end
  
htclose(fp);