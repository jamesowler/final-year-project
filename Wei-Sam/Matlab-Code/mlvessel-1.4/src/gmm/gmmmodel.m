function [gaussians, Q] = gmmmodel(samplematrix, k, type)
% function [gaussians, Q] = gmmmodel(samplematrix, k, type)
%
% Creates gaussianas for samples in "samplematrix" using EM. The type
% of covariance matrix is "type" and the number of gaussians is
% "k". Returns the gaussians in the form of a struct array. The vector
% returned Q contains the log-likelihood (of the present and missing,
% data, i.e. including the likelihood of the labels) of the training
% samples at each iteration and is used to verify convergence.
%
% "type" is a string. Can be:
%
% 'scalar'   - covariances are multiples of the identity matrix,
%              represented by a scalar.
% 'diagonal' - diagonal covariances matrices.
% 'full'     - no restriction on the covariance matrices.
% 'equal'    - all covariance matrices are equal.
% 
% See also: gmmcreatemodel.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializing:
%
% Random samples are chosen to be the gaussians means. Equal weights
% are given to each gaussian. The covariance matrices are initialized
% as the identity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nsamples nfeatures] = size(samplematrix);

if (k > nsamples)
  disp('Warning: k (number of gaussians) is larger than number of samples');
end

% Choosing random samples.
randindex = randperm(nsamples);
meanmatrix = samplematrix(randindex(1:k), :);
%scale = (max(samplematrix(:,1)) - min(samplematrix(:,1))) * 5e-2;
scale = 1000;

% Initializing covariances.
switch type
 case 'scalar'
  for i = 1:k
    gaussians(i).mean = transpose(meanmatrix(i, :));
    gaussians(i).weight = 1 / k;
    gaussians(i).covariance = 1 * scale;
  end
 case 'diagonal'
  for i = 1:k
    gaussians(i).mean = transpose(meanmatrix(i, :));
    gaussians(i).weight = 1 / k;
    gaussians(i).covariance = ones(1, nfeatures) * scale;
  end
 case 'full'
  for i = 1:k
    gaussians(i).mean = transpose(meanmatrix(i, :));
    gaussians(i).weight = 1 / k;
    gaussians(i).covariance = eye(nfeatures) * scale;
  end
 case 'equal'
  for i = 1:k
    gaussians(i).mean = transpose(meanmatrix(i, :));
    gaussians(i).weight = 1 / k;
  end
  equalcovariance = eye(nfeatures) * scale;
 otherwise
  disp('Error: wierd covariance type');
  return;
end

%
% Adjusting the algoritm's parameters.
%

epsilon = 1e-4;
%epsilon = 0;
T = epsilon * nsamples;

iterations = 0;
maxiterations = 40;

minvar = (1 / (10 * k))^2;

%dummies
Q(1) = -1e10;
difference = T + 1;

% COND(X) returns the 2-norm condition number (the ratio of the
%    largest singular value of X to the smallest).  Large condition
%    numbers indicate a nearly singular matrix
condlimit = 10^8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The EM algorithms       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch type
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Scalar variance          %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
 case 'scalar'
  
  % difference <= T means convergence
  while difference > T & iterations < maxiterations
    %drawscalar(samplematrix, gaussians);
    
    oldgaussians = gaussians;
    
    clear p;
    clear dists;
    
    % Calculating probabilities
    % First, caclulates the joint distribution for i (gaussian) and
    % x (feature vector).
    for i = 1:size(gaussians, 2)
      sigmaquadrado = oldgaussians(i).covariance;
      
      dists(i, :) = sum((samplematrix - repmat(oldgaussians(i).mean', ...
                                               [nsamples 1])).^2, 2)';

      factor = (2 * pi * sigmaquadrado)^(- nfeatures/2) * ...
          oldgaussians(i).weight;
      
      p(i, :) =  factor * exp(dists(i, :) / (- 2 * sigmaquadrado));
    end

    % Now, from the joint to posterior.
    expectedloglikelihood = 0;
    for j = 1:nsamples
      sampleprob = sum(p(:, j));
      
      if  sampleprob == 0
        disp('Warning: a sample had zero probability');
        % No gaussian contributes to the samples probability.
        % Making up probabilities for the sample.
        p(:, j) = 1/size(gaussians, 2);
      else
        joint = p(:, j);
        logjoint = zeros(size(joint));
        logjoint(joint > 0) = log(joint(joint > 0));

        p(:, j) = p(:, j) / sampleprob;
        expectedloglikelihood = expectedloglikelihood + dot(p(:, j), logjoint);
      end
    end
    
    % Calculating new parameters.
    for i = 1:size(gaussians, 2)
      somaprob = sum(p(i, :));
      
      gaussians(i).mean = ((p(i, :) * samplematrix) / somaprob)';

      gaussians(i).covariance = (p(i, :) * dists(i, :)') / (somaprob ...
                                                        * nfeatures);
            
      gaussians(i).weight = somaprob / nsamples;
      
      if (gaussians(i).covariance < minvar)
        disp('Warning: low variance.');
        gaussians(i).covariance = minvar;
      end

    end

    % Calculating how much the likelihood changed.
    Q(iterations + 2) = expectedloglikelihood;
    difference = abs(Q(iterations + 2) - Q(iterations + 1));

    iterations = iterations + 1;
  end

  disp('End of algorithm');
  disp(['Iterations executed: ' num2str(iterations)]);
  disp(['T used: ' num2str(T)]);

  %figure; drawscalar(samplematrix, gaussians);
  
  for i = 1:size(gaussians, 2)
    gaussians(i).covariance = eye(nfeatures) * gaussians(i).covariance;
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%
  % Diagonal covariances %
  %%%%%%%%%%%%%%%%%%%%%%%%
 case 'diagonal'

  % difference <= T means convergence
  while difference > T & iterations < maxiterations
    %drawdiagonal(samplematrix, gaussians);
     
    oldgaussians = gaussians;
    
    clear p;
    clear diff;

    % Calculating probabilities
    % First, caclulates the joint distribution for i (gaussian) and
    % x (feature vector).
    for i = 1:size(gaussians, 2)
      detcov = prod(oldgaussians(i).covariance);
      
      invcov = 1 ./ oldgaussians(i).covariance;
      
      diff{i} = (samplematrix - repmat(oldgaussians(i).mean', ...
                                       [nsamples 1])).^2;

      factor = (2 * pi)^( -nfeatures/2) * detcov^( -1/2) * ...
               oldgaussians(i).weight;
      
      p(i, :) = factor * exp( (-1/2) * invcov * diff{i}' );
    end
    
    % Now, from the joint to posterior.
    expectedloglikelihood = 0;
    for j = 1:nsamples
      sampleprob = sum(p(:, j));
      
      if  sampleprob == 0
        disp('Warning: a sample had zero probability');
        % No gaussian contributes to the samples probability.
        % Making up probabilities for the sample.
        p(:, j) = 1/size(gaussians, 2);
      else
        joint = p(:, j);
        logjoint = zeros(size(joint));
        logjoint(joint > 0) = log(joint(joint > 0));

        p(:, j) = p(:, j) / sampleprob;
        expectedloglikelihood = expectedloglikelihood + dot(p(:, j), logjoint);
      end
    
    end
    
    % Calculating new parameters.
    for i = 1:size(gaussians, 2)
      somaprob = sum(p(i, :));
      
      gaussians(i).mean = transpose(((p(i, :) * samplematrix) / somaprob));

      gaussians(i).covariance = (p(i, :) * diff{i}) / somaprob;
            
      gaussians(i).weight = somaprob / nsamples;
      
      for d = 1:nfeatures
        if (gaussians(i).covariance(d) < minvar)
          disp('Warning: low variance.');
          gaussians(i).covariance(d) = minvar;
        end
      end
    end

    % Calculating how much the likelihood changed.
    Q(iterations + 2) = expectedloglikelihood;
    difference = abs(Q(iterations + 2) - Q(iterations + 1));

    iterations = iterations + 1;
  end
  
  disp('End of algorithm');
  disp(['Iterations executed: ' num2str(iterations)]);
  disp(['T used: ' num2str(T)]);
  
  %figure; drawdiagonal(samplematrix, gaussians);
  
  for i = 1:size(gaussians, 2)
    gaussians(i).covariance =  diag(gaussians(i).covariance);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%
  % Complete covariances  %
  %%%%%%%%%%%%%%%%%%%%%%%%%
  
 case 'full'
  
  % difference <= T means convergence
  while difference > T & iterations < maxiterations
    %drawfull(samplematrix, gaussians);
    
    oldgaussians = gaussians;
    
    clear p;
    clear diff;
    
    % Calculating probabilities
    % First, caclulates the joint distribution for i (gaussian) and
    % x (feature vector).
    for i = 1:size(gaussians, 2)
      detcov = det(oldgaussians(i).covariance);
      
      invcov =  inv(oldgaussians(i).covariance);
      
      diff{i} = samplematrix - repmat(oldgaussians(i).mean', [nsamples 1]);

      factor = (2 * pi)^(-nfeatures/2) * detcov^(-1/2) * ...
               oldgaussians(i).weight;

      p(i, :) = factor * exp( (-1/2) * sum(diff{i} * invcov .* diff{i}, 2)' );
    end

    % Now, from the joint to posterior.
    expectedloglikelihood = 0;
    for j = 1:nsamples
      sampleprob = sum(p(:, j));
      
      if  sampleprob == 0
        disp('Warning: a sample had zero probability');
        % No gaussian contributes to the samples probability.
        % Making up probabilities for the sample.
        p(:, j) = 1/size(gaussians, 2);
      else
        joint = p(:, j);
        logjoint = zeros(size(joint));
        logjoint(joint > 0) = log(joint(joint > 0));

        p(:, j) = p(:, j) / sampleprob;
        expectedloglikelihood = expectedloglikelihood + dot(p(:, j), logjoint);
      end
    
    end
    
    % Calculating new parameters.
    for i = 1:size(gaussians, 2)
      somaprob = sum(p(i, :));
      
      gaussians(i).mean = ((p(i, :) * samplematrix) / somaprob)';

      gaussians(i).weight = somaprob / nsamples;

      cov = zeros(nfeatures);
      for k = 1:nsamples
        delta = diff{i}(k, :);
        cov = cov + p(i, k) * delta' * delta;
      end
      gaussians(i).covariance = cov / somaprob;
    end

    % Calculating how much the likelihood changed.
    Q(iterations + 2) = expectedloglikelihood;
    difference = abs(Q(iterations + 2) - Q(iterations + 1));

    iterations = iterations + 1;

    % Eliminates gaussianas with high covariance.
    i = 1;
    while i <= size(gaussians, 2)
      if (cond(gaussians(i).covariance) > condlimit)
        disp(['Warning: very high condition number: ' ...
              'gaussian eliminated.']);
        disp(['Condition number: ' num2str(cond(gaussians(i).covariance))]);
        if i < size(gaussians, 2)
          gaussians = [gaussians(1:i-1) gaussians(i+1:end)];
        else
          gaussians = gaussians(1:i-1);
        end
        i = i - 1;
      end
      i = i + 1;
    end
    
  end

  disp('End of algorithm');
  disp(['Iterations executed: ' num2str(iterations)]);
  disp(['T used: ' num2str(T)]);
  
  %figure; drawfull(samplematrix, gaussians);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Complete and equal covariances %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
 case 'equal'
  
  % difference <= T means convergence
  while difference > T & iterations < maxiterations
    %drawequal(samplematrix, gaussians, equalcovariance);
    
    oldcovariance = equalcovariance;
    oldgaussians = gaussians;
    
    clear p;
    clear diff;
    
    % Calculating probabilities
    % First, caclulates the joint distribution for i (gaussian) and
    % x (feature vector).
    detcov = det(oldcovariance);
    
    invcov =  inv(oldcovariance);
    for i = 1:size(gaussians, 2)
      
      diff{i} = samplematrix - repmat(oldgaussians(i).mean', [nsamples 1]);

      factor = (2 * pi)^(-nfeatures/2) * detcov^(-1/2) * ...
               oldgaussians(i).weight;
      
      p(i, :) = factor * exp( (-1/2) * sum(diff{i} * invcov .* diff{i}, 2)' );
    end
    
    % Now, from the joint to posterior.
    expectedloglikelihood = 0;
    for j = 1:nsamples
      sampleprob = sum(p(:, j));
      
      if  sampleprob == 0
        disp('Warning: a sample had zero probability');
        % No gaussian contributes to the samples probability.
        % Making up probabilities for the sample.
        p(:, j) = 1/size(gaussians, 2);
      else
        joint = p(:, j);
        logjoint = zeros(size(joint));
        logjoint(joint > 0) = log(joint(joint > 0));

        p(:, j) = p(:, j) / sampleprob;
        expectedloglikelihood = expectedloglikelihood + dot(p(:, j), logjoint);
      end
    
    end
    
    % Calculating new parameters.
    equalcovariance = zeros(nfeatures);
    
    for i = 1:size(gaussians, 2)
      somaprob = sum(p(i, :));
      
      gaussians(i).mean = ((p(i, :) * samplematrix) / somaprob)';
      
      gaussians(i).weight = somaprob / nsamples;
      
      cov = zeros(nfeatures);
      for k = 1:nsamples
        delta = diff{i}(k, :);
        cov = cov + p(i, k) * delta' * delta;
      end
      equalcovariance = equalcovariance + gaussians(i).weight * (cov ...
                                                        / somaprob);
    end
    
    % Warns if covariance is wierd.
    if (cond(gaussians(i).covariance) > condlimit)
      disp('Warning: covariance matrix with high condition number.');
    end
    
    % Calculating how much the likelihood changed.
    Q(iterations + 2) = expectedloglikelihood;
    difference = abs(Q(iterations + 2) - Q(iterations + 1));
    
    iterations = iterations + 1;
    
  end

  disp('End of algorithm');
  disp(['Iterations executed: ' num2str(iterations)]);
  disp(['T used: ' num2str(T)]);
  
  %figure; drawequal(samplematrix, gaussians, equalcovariance);
  
  for i = 1:size(gaussians, 2)
    gaussians(i).covariance =  equalcovariance;
  end
end

% Removes dummy from vector
Q = Q(2:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Drawing the gaussians.
function drawscalar(samplematrix, gaussians)

f1 = samplematrix(:,1); 
f2 = samplematrix(:,2);

xmin = min(f1);
xmax = max(f1);

ymin = min(f2);
ymax = max(f2);

dense = plotdense(f1, f2, [xmin xmax ymin ymax], [1000 1000]);

maxcount = max(dense(:));

clf;
axes;
hold on;
iptsetpref('ImshowAxesVisible', 'on');

imshow([xmin xmax], [ymin ymax], 1 - (log(dense + 1) / ...
                                      log(maxcount + 1)));

hold on;

for i = 1:size(gaussians, 2)
  m1 = gaussians(i).mean(1);
  m2 = gaussians(i).mean(2);
  
  s1 = gaussians(i).covariance;
  s2 = gaussians(i).covariance;
  
  text(m1, m2, int2str(i), 'FontSize', 14, 'HorizontalAlignment', 'center');
  
  circle(m1,m2,s1,s2, 'k-');
end

hold off;

input('Press enter', 's');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function drawdiagonal(samplematrix, gaussians)

f1 = samplematrix(:,1); 
f2 = samplematrix(:,2);

xmin = min(f1);
xmax = max(f1);

ymin = min(f2);
ymax = max(f2);

dense = plotdense(f1, f2, [xmin xmax ymin ymax], [1000 1000]);

maxcount = max(dense(:));
  
clf;
axes;
hold on;
iptsetpref('ImshowAxesVisible', 'on');

imshow([xmin xmax], [ymin ymax], 1 - (log(dense + 1) / ...
                                      log(maxcount + 1)));

hold on;

for i = 1:size(gaussians, 2)
  m1 = gaussians(i).mean(1);
  m2 = gaussians(i).mean(2);
  
  s1 = gaussians(i).covariance(1);
  s2 = gaussians(i).covariance(2);
  
  text(m1, m2, int2str(i), 'FontSize', 14, 'HorizontalAlignment', 'center');
  
  circle(m1,m2,s1,s2, 'k-');
end

hold off;

input('Press enter', 's');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function drawfull(samplematrix, gaussians);

f1 = samplematrix(:,1); 
f2 = samplematrix(:,2);

xmin = min(f1);
xmax = max(f1);

ymin = min(f2);
ymax = max(f2);

dense = plotdense(f1, f2, [xmin xmax ymin ymax], [1000 1000]);

maxcount = max(dense(:));

clf;
axes;
hold on;
iptsetpref('ImshowAxesVisible', 'on');

imshow([xmin xmax], [ymin ymax], 1 - (log(dense + 1) / ...
                                      log(maxcount + 1)));

hold on;

for i = 1:size(gaussians, 2)
  meani = gaussians(i).mean(1:2);
    
  cov = gaussians(i).covariance(1:2, 1:2);
  
  text(meani(1), meani(2), int2str(i), 'FontSize', 14, ...
       'HorizontalAlignment', 'center');
  
  ellipse(meani,cov, 'k-');
end

hold off;

input('Press enter', 's');
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function drawequal(samplematrix, gaussians, equalcovariance);

f1 = samplematrix(:,1); 
f2 = samplematrix(:,2);

xmin = min(f1);
xmax = max(f1);

ymin = min(f2);
ymax = max(f2);

dense = plotdense(f1, f2, [xmin xmax ymin ymax], [1000 1000]);

maxcount = max(dense(:));

clf;
axes;
hold on;
iptsetpref('ImshowAxesVisible', 'on');

imshow([xmin xmax], [ymin ymax], 1 - (log(dense + 1) / ...
                                      log(maxcount + 1)));

hold on;

cov = equalcovariance(1:2, 1:2);

for i = 1:size(gaussians, 2)
  meani = gaussians(i).mean(1:2);
  
  text(meani(1), meani(2), int2str(i), 'FontSize', 14, ...
       'HorizontalAlignment', 'center');
  
  ellipse(meani, cov, 'k-');
end

hold off;

input('Press enter', 's');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ellipse(emean, cov, style)

theta = linspace(0, 2*pi, 100);

[V, D] = eig(cov);

eigvectors(:, 1) = V(:, 1) / norm(V(:, 1));
eigvectors(:, 2) = V(:, 2) / norm(V(:, 2));

eigvectors = eigvectors * D;

x = emean(1) + ((cos(theta) * eigvectors(1, 1)) + (sin(theta) * ...
                                                  eigvectors(1, 2)));
y = emean(2) + ((cos(theta) * eigvectors(2, 1)) + (sin(theta) * ...
                                                  eigvectors(2, 2)));

if (nargin == 4)
  style = 'k-.';
end

plot(x, y, style);