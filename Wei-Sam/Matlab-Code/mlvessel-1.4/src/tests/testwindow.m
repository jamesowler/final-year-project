function testwindow(config)
% testwindow(config)
%
% Creates results training with a window of the same image being
% segmented. All the configuration parameters and image names should
% be informed in "config".
%
% See also: testleaveoneout, testmixed, stareconfig, driveconfig.

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

% Should window test be done under this configuration?
if (~config.window)
  return;
end

% The stats generation module.
addpath([installdir filesep 'src' filesep 'stats']);
% The gaussian mixture model module.
addpath([installdir filesep 'src' filesep 'gmm']);
% The linear classifier module.
addpath([installdir filesep 'src' filesep 'lmse']);
% The knn classifier module.
addpath([installdir filesep 'src' filesep 'knn']);
% The skeletonization module.
addpath([installdir filesep 'src' filesep 'skel']);
% The feature manipulation module.
addpath([installdir filesep 'src' filesep 'ftrs']);
% The html module.
addpath([installdir filesep 'src' filesep 'html']);

% Directory with images.
addpath(config.imagedir);

% Creating directory for results.
resultdir = config.resultdir;
testname = ['window_' config.testname];
ignore = mkdir(resultdir, testname);
resultdir = [resultdir testname filesep];

% Loop to test each image separatly.
for ind = 1:size(config.windownames, 2)
  name = config.windownames{ind}.original
  manualname = config.windownames{ind}.manual1;

  shortname = name(1:(end-4));
  outputdir = [resultdir shortname filesep];
  ignore = mkdir(resultdir, shortname);

  window_test(outputdir, name, manualname, config);
end

% Creates stats and puts them on a web page.
if (config.createstats)
  createstats(resultdir, config.windownames, config);
  createstatspage(resultdir, config.windownames);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function window_test(outputdir, name, manualname, config)
%
% Performs the test for one image, saving results in html format.
%  

% Filenames
shortname = name(1:(end-4));
featurefilename = [config.featuredir shortname '-features.mat'];
labelledfilename = [config.labelleddir shortname '-roi-labelled.mat'];

%Creating the features file from the image and labelled file from
%the feature file and the manual segmentation.
img = imread(name);

disp('Creating features file.');
switch(config.imagetype)
 case 'angiogram'
  createfeaturesangiogram(img, config.morletscales, config.paderosionsize, ...
                          featurefilename);
 case 'redfree'
  createfeaturesredfree(img, config.morletscales, config.paderosionsize, ...
                        featurefilename);
 case 'colored'
  createfeaturescolored(img, config.morletscales, ...
                        config.paderosionsize, featurefilename);
 otherwise
  disp(['Unknown image type ' config.imagetype]);
  return;
end

disp('Creating labelled file');
switch(config.manualtype)
 case 'segmentation'
  roi = createlabelled(featurefilename, manualname, labelledfilename, ...
                       config.windowsize);
 case 'skeleton'
  roi = createlabelledskeleton(featurefilename, manualname, ...
                               labelledfilename, config.windowsize);
 otherwise
  disp(['Unknown manual image type ' config.manualtype]);
  return;
end

% Creating the processed samples file from the labelled samples file.
trainingsamples = config.trainingsamples;
processedfilename = [config.processeddir shortname '-roi-processed-' ...
                    num2str(trainingsamples) '.mat'];
disp('Creating processed file');
createprocessed(labelledfilename, processedfilename, trainingsamples);

switch(config.classifier)
 case 'gmm'
  % Gaussian model paremeters
  ngaussians = config.ngaussians;
  covariancetype = config.covariancetype;
  modelfilename = [config.modeldir shortname '-roi-' covariancetype ...
                   '-' num2str(ngaussians) '.mat'];

  % Creating the gaussian mixture model from the processed labelled
  % samples file.
  disp('Creating gaussian mixture models');
  gmmcreatemodel(processedfilename, modelfilename, ngaussians, ...
                 ngaussians, covariancetype);
  
  % Creates the posterior probabilities image by pixel classification.
  [vessellikelihoods, restlikelihoods, vesselprior, restprior] = ...
      gmmlikelihoods(featurefilename, modelfilename, config.normalizeflag);
  
  vesselprob = vessellikelihoods * vesselprior;
  restprob = restlikelihoods * restprior;
  
  % Posterior probabilities.
  classgray = (vesselprob) ./ (restprob + vesselprob);
  classgray((vesselprob + restprob) == 0) = 0;
  
  % The segmentation.
  class = classgray > 0.5;
  
  classgray(classgray < 0) = 0;
  classgrayeval = (((classgray - 0.5).^3 / (0.5)^3) + 1) / 2;
  
  classifierfilename = modelfilename;
 case 'lmse'
  disp('Creating linear function.');
  linearfilename = [config.lineardir shortname '-roi-lmse.mat'];
  lmsecreatelinear(processedfilename, linearfilename);
  [class, classgray] = lmseapply(featurefilename, linearfilename, ...
                                 config.normalizeflag);
  
    
  classgrayeval = (classgray + 2) / 4;
  classgray = (classgray + 1) / 2;

  classgrayeval(classgrayeval < 0) = 0;
  classgray(classgray < 0) = 0;
  
  classifierfilename = linearfilename;
 case 'knn'
  disp('Classifying with knn.');
  [class, classgray] = knnclassify(featurefilename, processedfilename, ...
                                   config.k, config.normalizeflag);

  classgray(classgray < 0) = 0;
  classgrayeval = (((classgray - 0.5).^3 / (0.5)^3) + 1) / 2;
  
  classifierfilename = [];
 otherwise
  disp(['Unknown classifier type ' config.classifier]);
  return;
end

% Extracts the skeleton from the segmentation.
[skel, final] = skeleton(class, 0, img);

% Creates the page and images.
fp = htopen([outputdir 'index.html'], ['Retina Image Processing ' ...
                    'for ' name]);
htwtext(fp,['<h2> Retina Image Processing for ' name '</h2>']); 

if (~isempty(classifierfilename))
  C = load(classifierfilename);
  
  % Info on classifier.
  htwtext(fp, 'About the classifier used: ');
  htwtext(fp, transpose(C.info));
  htwtext(fp, '<br>Features used:<br>');
  htwtext(fp, transpose(C.description));
  
  if (strcmp(config.classifier, 'gmm'))
    handle = figure('visible', 'off');
    
    htwtext(fp, '<h4>Log-likelihood of vessel samples during EM</h4>');
    htwtext(fp, '<img src="vesselq.jpg" width = 25%>'); 
    plot(C.vesselQ);
    print(handle, '-djpeg', [outputdir filesep 'vesselq.jpg']);

    htwtext(fp, '<h4>Log-likelihood of rest samples during EM</h4>');
    htwtext(fp, '<img src="restq.jpg" width = 25%>'); 
    plot(C.restQ);
    print(handle, '-djpeg', [outputdir filesep 'restq.jpg']);
  end
  
  clear C;
end
  
% Original image.
htwpar(fp);htwhr(fp); htwpar(fp);
httitle='Original image';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, img, [shortname '.jpg'], 4);

manskel = imread(manualname);
manskel = manskel > 128;

% Region of interest used for training.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Region of pixels used for training';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, roi, [shortname '-roi.png'], 4);
htwimage(fp, outputdir, (roi & manskel), [shortname '-roi-final.png'], 4);

% Posterior probabilities and transform to be evaluated.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Posterior probabilities image and its transform';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, classgray, [shortname '-class-gray.png'], 4);
htwimage(fp, outputdir, classgrayeval, [shortname '-class-gray-eval.png'], 4);

% Manual and automatic segmentations.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Manual and automatic segmentations';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, manskel, [shortname '-manual-class.png'], 4);
htwimage(fp, outputdir, class, [shortname '-class.png'], 4);

% Manual and automatic skeletons.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Manual and automatic skeletons';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, bwmorph(manskel, 'skel', Inf),...
              [shortname '-manual-skeleton.png'], 4);
htwimage(fp, outputdir, skel, [shortname '-class-skeleton.png'], 4);

% Skeleton superposed on image.
htwpar(fp); htwhr(fp); htwpar(fp);
httitle='Skeleton of segmentation produced by classifier with image';
htwtext(fp, ['<h2>' httitle '</h2>']);
htwimage(fp, outputdir, final, [shortname '-class-final.jpg'], 4);

htclose(fp);
