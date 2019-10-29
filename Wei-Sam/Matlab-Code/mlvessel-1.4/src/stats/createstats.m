function createstats(resultdir, imagenames, config)
% createstats(resultdir, imagenames, config)
%
% Creates and saves some statistics on image results: accuracy, ROC
% curves (as well as curve plots), and area under the ROC curves.
%
% The statistics are created based on the results in "resultdir", for
% the images indicated in the cell array "imagenames". "config"
% contains configuration data about the test being perform, such as in
% "driveconfig" and "stareconfig". The results being analysed should
% have been created with test functions such as "testmixed" and
% "testleaveoneout".
% 
% If there is a second set of manual segmentations, as indicated in
% "config", its ROC data and accuracy are calculated as well. A second
% set of apertures (besides the automatic one) can also be specified
% through "config".
%
% See also: createstatspage, testmixed, testleaveoneout, driveconfig,
%           stareconfig, accuracy, rocdata, rocarea, plotroc.

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

% Creates statistical data for each image.
for i = 1:size(imagenames, 2)
  imagename = imagenames{i};
  imfilename = imagename.original
  
  %Creating apertures.
  apertures = [];
  
  % Open file aperture if exists.
  if (config.otherapertures)
    ap = imread(imagename.aperture);
    if (islogical(ap))
      aperture.img = ap;
    else
      aperture.img = ap > 128;
    end
      
    aperture.description = 'file';
    apertures = [apertures, aperture];
  end
  
  % Automatic aperture.
  switch(config.imagetype)
   case 'angiogram'
    ap = createretinamaskangiogram(imread(imfilename));
   case 'redfree'
    ap = createretinamaskredfree(imread(imfilename));
   case 'colored'
    ap = createretinamaskcolored(imread(imfilename));
  end
 
  aperture.img = ap;
  aperture.description = 'auto';
  apertures = [apertures, aperture];

  % Loading ground truth.
  control = imread(imagename.manual1);
  if (max(control(:)) > 128)
      control = control > 128;
  else
    control = control > 0.5;
  end

  % Loading gray-level result.
  grayimage = imread([resultdir filesep imfilename(1:(end-4)) filesep ...
                      imfilename(1:(end-4)) '-class-gray-eval.png']);
  grayimage = double(grayimage);

  % Loading classification result.
  testimage = logical(imread([resultdir filesep imfilename(1:(end-4)) filesep ...
                      imfilename(1:(end-4)) '-class.png'])); %ADDED 'logical() 14/12/2017
  
  % Creating stats for each aperture.
  stats = [];
  description = {};
  for a = 1:size(apertures, 2)
    aperture = apertures(a);
    ap = aperture.img;
    
    % ROC data.
    [autofp(a,i,:), autotp(a,i,:), negative(a,i), positive(a,i)] = ...
        rocdata(control, grayimage, ap, 0:256);
    
    % Accuracy.
    [autoac, autotrue(a,i), total(a,i)] = accuracy(control, testimage, ap);
    
    % Area under ROC.
    az = rocarea(squeeze(autofp(a,i,:)), squeeze(autotp(a,i,:)), ...
                 negative(a,i), positive(a,i));
    
    stats = [stats autoac az];
    description = {description{:},...
                   ['method accuracy (' aperture.description ' aperture)'],...
                   ['method ROC area (' aperture.description ' aperture)']};
    
    % Creating manual segmentation's accuracy and ROC data.
    if (config.secondmanual)
      manual = imread(imagename.manual2);
      if (max(manual(:)) > 128)
          manual = double(manual > 128);
      else
          manual = double(manual > 0.5);
      end
      
      [manfp(a,i,:), mantp(a,i,:)] = rocdata(control, manual, ap, 1);
      
      [manac, mantrue(a,i)] = accuracy(control, manual, ap);
      
      stats = [stats manac];
      
      description = {description{:},...
                     ['manaul accuracy (' aperture.description ' aperture)']};
    end
  end
  
  % Saving the stats.
  outputfilename = [resultdir filesep imfilename(1:(end-4)) filesep ...
                    'stats.mat'];
  save(outputfilename, 'description', 'stats', 'imfilename', '-MAT');

  % Saving ROC data.
  tp = {};
  fp = {};
  p =[];
  n = [];
  description = {};
  for a = 1:size(apertures, 2)
    aperture = apertures(a);
    
    tp = {tp{:}, squeeze(autotp(a,i,:))};
    fp = {fp{:}, squeeze(autofp(a,i,:))};
    
    p = [p, positive(a,i)];
    n = [n, negative(a,i)];
    
    description = {description{:}, ...
                   ['method roc (' aperture.description ' aperture)']};

    % Specifying the p > 0.5 point
    tp = {tp{:}, squeeze(autotp(a,i,128))};
    fp = {fp{:}, squeeze(autofp(a,i,128))};
    
    p = [p, positive(a,i)];
    n = [n, negative(a,i)];
    
    description = {description{:}, ...
                   ['method p > 0.5 (' aperture.description ' aperture)']};
    
    % Adding manual ROC data.
    if (config.secondmanual)
      tp = {tp{:}, mantp(a,i)};
      fp = {fp{:}, manfp(a,i)};
      
      p = [p, positive(a,i)];
      n = [n, negative(a,i)];
      
      description = {description{:},...
                     ['manaul roc (' aperture.description ' aperture)']};
    end
  end
  
  rocfilename = [resultdir filesep imfilename(1:(end-4)) filesep 'rocs.mat'];
  save(rocfilename, 'description', 'tp', 'fp', 'p', 'n', 'imfilename', '-MAT');
  
  % Saving ROC graphs.
  handle = plotroc({rocfilename});
  
  print(handle, '-djpeg', [resultdir filesep imfilename(1:(end-4)) ...
                      filesep 'rocs.jpg']);
  print(handle, '-deps', [resultdir filesep imfilename(1:(end-4)) ...
                      filesep 'rocs.eps']);
end

% Averaging statistics over all images.

% Saving stats.
stats = [];
description = {};
for a = 1:size(apertures, 2)
  aperture = apertures(a);
  
  % Accuracy
  autoac = sum(autotrue(a,:)) / sum(total(a,:));
  
  % Area under roc
  az = rocarea(squeeze(sum(autofp(a,:,:), 2)), ...
               squeeze(sum(autotp(a,:,:), 2)), ...
               sum(negative(a,:)), sum(positive(a,:)));
  
  stats = [stats autoac az];
  description = {description{:},...
                 ['method accuracy (' aperture.description ' aperture)'],...
                 ['method ROC area (' aperture.description ' aperture)']};

  % Manual accuracy.
  if(config.secondmanual)
    manac = sum(mantrue(a,:)) / sum(total(a,:));
    
    stats = [stats manac];
    description = {description{:},...
                   ['manaul accuracy (' aperture.description ' aperture)']};
  end
end

outputfilename = [resultdir filesep 'stats.mat'];
save(outputfilename, 'description', 'stats', '-MAT');

% Saving ROC data
tp = {};
fp = {};
p =[];
n = [];
description = {};

for a = 1:size(apertures, 2)
  aperture = apertures(a);

  tp = {tp{:}, squeeze(sum(autotp(a,:,:), 2))};
  fp = {fp{:}, squeeze(sum(autofp(a,:,:), 2))};
  
  p = [p, sum(positive(a,:))];
  n = [n, sum(negative(a,:))];
  
  description = {description{:},...
                 ['method roc (' aperture.description ' aperture)']};

  % Specifying the p > 0.5 point
  tp = {tp{:}, squeeze(sum(autotp(a,:,128)))};
  fp = {fp{:}, squeeze(sum(autofp(a,:,128)))};
  
  p = [p, sum(positive(a,:))];
  n = [n, sum(negative(a,:))];
  
  description = {description{:}, ...
                 ['method p > 0.5 (' aperture.description ' aperture)']};
  
  if (config.secondmanual)
    tp = {tp{:}, sum(mantp(a,:))};
    fp = {fp{:}, sum(manfp(a,:))};
    
    p = [p, sum(positive(a,:))];
    n = [n, sum(negative(a,:))];
    
    description = {description{:},...
                   ['manaul roc (' aperture.description ' aperture)']};
  end
end

rocfilename = [resultdir filesep 'rocs.mat'];
save(rocfilename, 'description', 'tp', 'fp', 'p', 'n', '-MAT');
  
% Saving ROC graphs.
handle = plotroc({rocfilename});
  
print(handle, '-djpeg', [resultdir filesep 'rocs.jpg']);
print(handle, '-deps', [resultdir filesep 'rocs.eps']);