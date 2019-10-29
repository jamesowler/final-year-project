function gmmcreatemodel(processedfilename, modelfilename, kvessel, ...
                        krest, type)
% gmmcreatemodel(processedfilename, modelfilename, kvessel, krest, type)
%
% Creates the gaussian mixture models using labelled samples in
% "processedfilename". Saves results in "modelfilename". The model
% will probably have "kvessel" gaussians representing vessel pixel
% samples and "krest" gaussians representin "non-vessels". "type"
% specifies the type of covariance matrix for the gaussians.
%
% See also: createprocessed, gmmmodel.

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

T = load(processedfilename);

description = T.description;

samplemean = T.samplemean;

samplestd = T.samplestd;

% "vessels" indicates vessel pixels in featurematrix.
vessels = (T.featurematrix(:, 1) == 1);

% Matrix with vessel samples.
vesselmatrix = T.featurematrix(vessels, 2:end);
% Matrix with non-vessel samples.
restmatrix = T.featurematrix(~vessels, 2:end);

% Number of vessel and non-vessel samples.
nvessel = size(vesselmatrix, 1);
nrest = size(restmatrix, 1);
ntotal = nvessel + nrest;

% Creating the mixture model for the vessels.
[vesselgaussians, vesselQ] = gmmmodel(vesselmatrix, kvessel, type);

% Creating the mixture model for non-vessels.
[restgaussians, restQ] = gmmmodel(restmatrix, krest, type);

% Priors.
vesselprior = nvessel / ntotal;
restprior = nrest / ntotal;

vesselit = size(vesselQ, 2);
restit = size(restQ, 2);

% Saves some info on parameters.
info = strvcat([' File used for training: ' processedfilename],...
               [' kvessel = ' num2str(kvessel)],...
               [' krest = ' num2str(krest)],...
               [' Iterations for vessels: ' num2str(vesselit)],...
               [' Iterations for rest: ' num2str(restit)],...
               [' Covariance matrix type: ' type]);

% Saves in "modelfilename" in matlab format.
save(modelfilename, 'description', 'info', 'vesselprior', ...
     'vesselgaussians', 'vesselQ', 'restprior', 'restgaussians', ...
     'restQ', 'samplemean', 'samplestd', '-MAT');