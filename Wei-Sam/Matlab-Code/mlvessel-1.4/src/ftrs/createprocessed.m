function createprocessed(labelledfilenames, processedfilename, nsamples)
% createprocessed(labelledfilenames, processedfilename, nsamples)
%
% Receives a two-dimensional array "labelledfilenames" with the names
% of labelles files containg labelled pixels along with their
% features. Creates "processedfilename", used for training
% classifiers. "nsamples" random samples are selected among all of the
% available.

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

featurematrix = [];
samples = [];
means = [];
vars = [];

nfiles = size(labelledfilenames, 1);
% Will randomly select nsamplesperfile from each file.
nsamplesperfile = ceil(nsamples / nfiles);

for i = 1:nfiles
  L = load(deblank(labelledfilenames(i,:)));

  [rows, columns] = size(L.featurematrix);
  if (nargin == 3 & nsamplesperfile < rows & nsamples ~= 0)
    % Selecting the random samples from the file.
    
    factor = nsamplesperfile/rows;

    vesselmatrix = L.featurematrix(L.featurematrix(:, 1) == 1, :);
    restmatrix = L.featurematrix(L.featurematrix(:, 1) == 0, :);
        
    nvessel = size(vesselmatrix, 1);
    nrest = size(restmatrix, 1);

    randindex = randperm(nvessel);
    vesselmatrix = vesselmatrix(randindex(1:ceil(nvessel * factor)), :);
    
    randindex = randperm(nrest);
    restmatrix = restmatrix(randindex(1:ceil(nrest * factor)), :);
        
    addmatrix = [vesselmatrix; restmatrix];
  else % preserves all samples
    addmatrix = L.featurematrix;
  end

  % Normalizes each feature matrix individually.
  m = mean(addmatrix(:, 2:end));
  var0 = var(addmatrix(:, 2:end));
  var1 = var(addmatrix(:, 2:end), 1);
  addmatrix(:, 2:end) = normfeats(addmatrix(:, 2:end), m, sqrt(var0));

  % Accumulates each set of sample's means, vars, and count.
  samples = [samples; size(addmatrix, 1)];
  means = [means; m];
  vars = [vars; var1];

  % Adds the new samples.
  featurematrix = [featurematrix; addmatrix];
end

description = L.description;

% Calculates the total means and stds from each image's.
samplemean = (samples' * means) / sum(samples);
samplevar = (samples' * vars) / (sum(samples) - 1);
samplestd = sqrt(samplevar);

% Saves in "processed" in matlab format.
save(processedfilename, 'description', 'featurematrix', 'samplemean', 'samplestd', '-MAT');