function config = stareconfig()

% Type of image: angiogram, redfree, or colored.
config.imagetype = 'colored';

% Scales for feature generation.
config.morletscales = [2 3 4 5];

% Erosion performed pefore creating artificial border around image
% (extending the image by padding).
config.paderosionsize = 5;

% We put everything under this directory.
rootdir = installdir;

% Gaussian model classifier and parameters.
% config.classifier = 'gmm';
% config.ngaussians = 20;
% config.trainingsamples = 1000000;
% config.covariancetype = 'full';
% config.modeldir     = [rootdir filesep 'models' filesep];
% % "normalizeflag" is 0 for same normalization on training and test set
% % and 1 otherwise.
% config.normalizeflag = 1;

%% Linear classifier and paremeters.
config.classifier = 'lmse';
config.trainingsamples = 1000000;
config.lineardir     = [rootdir filesep 'linear' filesep];
% "normalizeflag" is 0 for same normalization on training and test set
% and 1 otherwise.
config.normalizeflag = 1;

% K-nearest neighbor classifier and parameters.
%config.classifier = 'knn';
%config.trainingsamples = 500;
%config.k = 5;
% % "normalizeflag" is 0 for same normalization on training and test set
% % and 1 otherwise.
%config.normalizeflag = 1;

% segmentation or skeleton? (for labelling)
config.manualtype = 'segmentation';

% size of window sides for training and segmenting same image.
config.windowsize = 0.4;

% Stats options.
% Should stats be created?
config.createstats = 1;

% Use second set of manual segmentations for stats generation?
config.secondmanual = 1;

% Use another set of apertures for stats generation?
config.otherapertures = 0;

% Output.
config.testname = 'stare_lmse';
config.resultdir = [rootdir filesep 'results' filesep];

config.featuredir   = [rootdir filesep 'features' filesep];
config.labelleddir  = [rootdir filesep 'features-labelled' filesep];
config.processeddir = [rootdir filesep 'features-processed' filesep];

% Input.
config.imagedir =   [rootdir filesep 'stare_images' filesep];

names = starenames;

% For separate training and test sets.
config.mixed = 0;
config.trainnames = {names{1:5}};
config.testnames = {names{6:end}};

% For the window test.
config.window = 1;
config.windownames = names;

% For leave-one-out tests.
config.leaveoneout = 1;
config.leaveoneoutnames = names;

%%
%% Specifies the file names for tests.
%%
function names = starenames()

ind = 0;

ind = ind+1;
names{ind}.original = 'im0001.ppm'; 
names{ind}.manual1 = 'im0001.ah.ppm';
names{ind}.manual2 = 'im0001.vk.ppm';
names{ind}.aperture = 'im0001.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0002.ppm';
names{ind}.manual1 = 'im0002.ah.ppm';
names{ind}.manual2 = 'im0002.vk.ppm';
names{ind}.aperture = 'im0002.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0003.ppm';
names{ind}.manual1 = 'im0003.ah.ppm';
names{ind}.manual2 = 'im0003.vk.ppm';
names{ind}.aperture = 'im0003.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0004.ppm';
names{ind}.manual1 = 'im0004.ah.ppm';
names{ind}.manual2 = 'im0004.vk.ppm';
names{ind}.aperture = 'im0004.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0005.ppm';
names{ind}.manual1 = 'im0005.ah.ppm';
names{ind}.manual2 = 'im0005.vk.ppm';
names{ind}.aperture = 'im0005.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0044.ppm';
names{ind}.manual1 = 'im0044.ah.ppm';
names{ind}.manual2 = 'im0044.vk.ppm';
names{ind}.aperture = 'im0044.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0077.ppm';
names{ind}.manual1 = 'im0077.ah.ppm';
names{ind}.manual2 = 'im0077.vk.ppm';
names{ind}.aperture = 'im0077.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0081.ppm';
names{ind}.manual1 = 'im0081.ah.ppm';
names{ind}.manual2 = 'im0081.vk.ppm';
names{ind}.aperture = 'im0081.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0082.ppm';
names{ind}.manual1 = 'im0082.ah.ppm';
names{ind}.manual2 = 'im0082.vk.ppm';
names{ind}.aperture = 'im0082.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0139.ppm';
names{ind}.manual1 = 'im0139.ah.ppm';
names{ind}.manual2 = 'im0139.vk.ppm';
names{ind}.aperture = 'im0139.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0162.ppm';
names{ind}.manual1 = 'im0162.ah.ppm';
names{ind}.manual2 = 'im0162.vk.ppm';
names{ind}.aperture = 'im0162.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0163.ppm';
names{ind}.manual1 = 'im0163.ah.ppm';
names{ind}.manual2 = 'im0163.vk.ppm';
names{ind}.aperture = 'im0163.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0235.ppm';
names{ind}.manual1 = 'im0235.ah.ppm';
names{ind}.manual2 = 'im0235.vk.ppm';
names{ind}.aperture = 'im0235.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0236.ppm';
names{ind}.manual1 = 'im0236.ah.ppm';
names{ind}.manual2 = 'im0236.vk.ppm';
names{ind}.aperture = 'im0236.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0239.ppm';
names{ind}.manual1 = 'im0239.ah.ppm';
names{ind}.manual2 = 'im0239.vk.ppm';
names{ind}.aperture = 'im0239.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0240.ppm';
names{ind}.manual1 = 'im0240.ah.ppm';
names{ind}.manual2 = 'im0240.vk.ppm';
names{ind}.aperture = 'im0240.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0255.ppm';
names{ind}.manual1 = 'im0255.ah.ppm';
names{ind}.manual2 = 'im0255.vk.ppm';
names{ind}.aperture = 'im0255.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0291.ppm';
names{ind}.manual1 = 'im0291.ah.ppm';
names{ind}.manual2 = 'im0291.vk.ppm';
names{ind}.aperture = 'im0291.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0319.ppm';
names{ind}.manual1 = 'im0319.ah.ppm';
names{ind}.manual2 = 'im0319.vk.ppm';
names{ind}.aperture = 'im0319.ap.ppm';

ind = ind+1;
names{ind}.original = 'im0324.ppm';
names{ind}.manual1 = 'im0324.ah.ppm';
names{ind}.manual2 = 'im0324.vk.ppm';
names{ind}.aperture = 'im0324.ap.ppm';
