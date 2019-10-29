function classifier = guicreateclassifier(config)
% classifier = guicreateclassifier(config)
%
% Creates and returns a classifier based on parameters in
% "config". For example of parameters, see "guiclassifier.m".

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

if ~isdeployed
  % The gaussian mixture model module.
  addpath([installdir filesep 'src' filesep 'gmm']);
  % The linear classifier module.
  addpath([installdir filesep 'src' filesep 'lmse']);
  % The knn classifier module.
  addpath([installdir filesep 'src' filesep 'knn']);
  % The feature manipulation module.
  addpath([installdir filesep 'src' filesep 'ftrs']);
end

classifier = config;

images = classifier.images;

tmpdir = [installdir filesep 'guitmp'];

labelledfilenames = [];
% Generate features and labelled files for all training images.
for i = 1:length(images)
  [path,shortname,ext,version] = fileparts(images{i}.imagefilename);
  img = imread(fullfile(images{i}.imagepathname, images{i}.imagefilename));
      
  featurefilename = [tmpdir filesep shortname '-features.mat'];
  labelledfilename = [tmpdir filesep shortname '-labelled.mat'];
  labelsfilename = fullfile(images{i}.labelspathname, images{i}.labelsfilename);
  
  disp(['Creating features for ' shortname]);
  guicreatefeatures(img, classifier.imagetype, classifier.features, ...
                    featurefilename);
  
  disp(['Creating labelled file for ' shortname]);
  createlabelled(featurefilename, labelsfilename, labelledfilename);

  % Deletes features file.
  delete(featurefilename);
  
  labelledfilenames = strvcat(labelledfilenames, labelledfilename);
end

% Generate processed file to train classifier.
disp('Creating file with training samples');
processedfilename = [tmpdir filesep 'processed.mat'];
switch classifier.sampling
    case 'all'
        samples = 0;
    case 'random'
        samples = classifier.samples;
end
createprocessed(labelledfilenames, processedfilename, samples);

% Delete labelled files.
for i = 1:size(labelledfilenames, 1)
    delete( deblank(labelledfilenames(i,:)) );
end

% Generate and train classifier.
switch classifier.type
  case 'gmm'
  % Creates gaussian mixture model classifier and retrieves data from
  % file.
  
  % Gaussian model parameters
  ngaussians = classifier.k;
  covariancetype = 'full';
  modelfilename = [tmpdir filesep 'gmmclassifier.mat'];

  % Creating the gaussian mixture model from the processed labelled
  % samples file.
  disp('Creating gaussian mixture models');
  gmmcreatemodel(processedfilename, modelfilename, ngaussians, ...
                 ngaussians, covariancetype);
  
  C = load(modelfilename);
  delete(modelfilename);
  % Saves data in classifier struct.
  classifier.info = C.info;
  classifier.vesselprior = C.vesselprior;
  classifier.restprior = C.restprior;
  classifier.vesselgaussians = C.vesselgaussians;
  classifier.restgaussians = C.restgaussians;
  classifier.vesselQ = C.vesselQ;
  classifier.samplemean = C.samplemean;
  classifier.samplestd = C.samplestd;
  
 case 'lmse'
  % Creates linear classifier and retrieves data from file.
  disp('Creating linear function.');
  linearfilename = [tmpdir filesep 'lmseclassifier.mat'];
  lmsecreatelinear(processedfilename, linearfilename);

  C = load(linearfilename);
  delete(linearfilename);

  classifier.info = C.info;
  classifier.X = C.X;
  classifier.th = C.th;
  classifier.samplemean = C.samplemean;
  classifier.samplestd = C.samplestd;

 case 'knn'
  % Retrieves processed data from file.
  P = load(processedfilename);
  
  classifier.featurematrix = P.featurematrix;
  classifier.samplemean = P.samplemean;
  classifier.samplestd = P.samplestd;
end

% Remove processed file.
delete(processedfilename);