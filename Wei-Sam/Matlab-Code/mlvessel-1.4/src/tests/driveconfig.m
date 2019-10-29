function config = driveconfig()

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
config.classifier = 'gmm';
config.ngaussians = 20;
config.trainingsamples = 1000000;
config.covariancetype = 'full';
config.modeldir = [rootdir filesep 'models' filesep];
% % "normalizeflag" is 0 for same normalization on training and test set
% % and 1 otherwise.
% config.normalizeflag = 1;

%% Linear classifier and paremeters.
%config.classifier = 'lmse';
%config.trainingsamples = 1000000;
%config.lineardir     = [rootdir filesep 'linear' filesep];

% "normalizeflag" is 0 for same normalization on training and test set
% and 1 otherwise.
config.normalizeflag = 1;

% K-nearest neighbor classifier and parameters.
%config.classifier = 'knn';
%config.trainingsamples = 1000;
%config.k = 50;
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
config.secondmanual = 0;

% Use another set of apertures for stats generation?
config.otherapertures = 1;

% Output.
config.testname = 'drive_gmm';
config.resultdir = [rootdir filesep 'results' filesep];

config.featuredir   = [rootdir filesep 'features' filesep];
config.labelleddir  = [rootdir filesep 'features-labelled' filesep];
config.processeddir = [rootdir filesep 'features-processed' filesep];

% Input.
config.imagedir =   [rootdir filesep 'drive_images' filesep];

names = drivenames;

% For separate training and test sets.
config.mixed = 1;
config.trainnames = {names{21:40}};
config.testnames = {names{1:20}};

% For the window test.
config.window = 1;
config.windownames = {names{1:20}};

% For leave-one-out tests.
config.leaveoneout = 0;
config.leaveoneoutnames = {names{1:20}};

%%
%% Specifies the file names for tests.
%%
function names = drivenames()

ind = 0;

ind = ind+1;
names{ind}.original = '01_test.tif'; 
names{ind}.manual1 = '01_manual1.gif';
names{ind}.manual2 = '01_manual2.gif';
names{ind}.aperture = '01_test_mask.gif';

ind = ind+1;
names{ind}.original = '02_test.tif'; 
names{ind}.manual1 = '02_manual1.gif';
names{ind}.manual2 = '02_manual2.gif';
names{ind}.aperture = '02_test_mask.gif';

ind = ind+1;
names{ind}.original = '03_test.tif'; 
names{ind}.manual1 = '03_manual1.gif';
names{ind}.manual2 = '03_manual2.gif';
names{ind}.aperture = '03_test_mask.gif';

ind = ind+1;
names{ind}.original = '04_test.tif'; 
names{ind}.manual1 = '04_manual1.gif';
names{ind}.manual2 = '04_manual2.gif';
names{ind}.aperture = '04_test_mask.gif';

ind = ind+1;
names{ind}.original = '05_test.tif'; 
names{ind}.manual1 = '05_manual1.gif';
names{ind}.manual2 = '05_manual2.gif';
names{ind}.aperture = '05_test_mask.gif';

ind = ind+1;
names{ind}.original = '06_test.tif'; 
names{ind}.manual1 = '06_manual1.gif';
names{ind}.manual2 = '06_manual2.gif';
names{ind}.aperture = '06_test_mask.gif';

ind = ind+1;
names{ind}.original = '07_test.tif'; 
names{ind}.manual1 = '07_manual1.gif';
names{ind}.manual2 = '07_manual2.gif';
names{ind}.aperture = '07_test_mask.gif';

ind = ind+1;
names{ind}.original = '08_test.tif'; 
names{ind}.manual1 = '08_manual1.gif';
names{ind}.manual2 = '08_manual2.gif';
names{ind}.aperture = '08_test_mask.gif';

ind = ind+1;
names{ind}.original = '09_test.tif'; 
names{ind}.manual1 = '09_manual1.gif';
names{ind}.manual2 = '09_manual2.gif';
names{ind}.aperture = '09_test_mask.gif';

ind = ind+1;
names{ind}.original = '10_test.tif'; 
names{ind}.manual1 = '10_manual1.gif';
names{ind}.manual2 = '10_manual2.gif';
names{ind}.aperture = '10_test_mask.gif';

ind = ind+1;
names{ind}.original = '11_test.tif'; 
names{ind}.manual1 = '11_manual1.gif';
names{ind}.manual2 = '11_manual2.gif';
names{ind}.aperture = '11_test_mask.gif';

ind = ind+1;
names{ind}.original = '12_test.tif'; 
names{ind}.manual1 = '12_manual1.gif';
names{ind}.manual2 = '12_manual2.gif';
names{ind}.aperture = '12_test_mask.gif';

ind = ind+1;
names{ind}.original = '13_test.tif'; 
names{ind}.manual1 = '13_manual1.gif';
names{ind}.manual2 = '13_manual2.gif';
names{ind}.aperture = '13_test_mask.gif';

ind = ind+1;
names{ind}.original = '14_test.tif'; 
names{ind}.manual1 = '14_manual1.gif';
names{ind}.manual2 = '14_manual2.gif';
names{ind}.aperture = '14_test_mask.gif';

ind = ind+1;
names{ind}.original = '15_test.tif'; 
names{ind}.manual1 = '15_manual1.gif';
names{ind}.manual2 = '15_manual2.gif';
names{ind}.aperture = '15_test_mask.gif';

ind = ind+1;
names{ind}.original = '16_test.tif'; 
names{ind}.manual1 = '16_manual1.gif';
names{ind}.manual2 = '16_manual2.gif';
names{ind}.aperture = '16_test_mask.gif';

ind = ind+1;
names{ind}.original = '17_test.tif'; 
names{ind}.manual1 = '17_manual1.gif';
names{ind}.manual2 = '17_manual2.gif';
names{ind}.aperture = '17_test_mask.gif';

ind = ind+1;
names{ind}.original = '18_test.tif'; 
names{ind}.manual1 = '18_manual1.gif';
names{ind}.manual2 = '18_manual2.gif';
names{ind}.aperture = '18_test_mask.gif';

ind = ind+1;
names{ind}.original = '19_test.tif'; 
names{ind}.manual1 = '19_manual1.gif';
names{ind}.manual2 = '19_manual2.gif';
names{ind}.aperture = '19_test_mask.gif';

ind = ind+1;
names{ind}.original = '20_test.tif'; 
names{ind}.manual1 = '20_manual1.gif';
names{ind}.manual2 = '20_manual2.gif';
names{ind}.aperture = '20_test_mask.gif';

ind = ind+1;
names{ind}.original = '21_training.tif'; 
names{ind}.manual1 = '21_manual1.gif';
names{ind}.aperture = '21_training_mask.gif';

ind = ind+1;
names{ind}.original = '22_training.tif'; 
names{ind}.manual1 = '22_manual1.gif';
names{ind}.aperture = '22_training_mask.gif';

ind = ind+1;
names{ind}.original = '23_training.tif'; 
names{ind}.manual1 = '23_manual1.gif';
names{ind}.aperture = '23_training_mask.gif';

ind = ind+1;
names{ind}.original = '24_training.tif'; 
names{ind}.manual1 = '24_manual1.gif';
names{ind}.aperture = '24_training_mask.gif';

ind = ind+1;
names{ind}.original = '25_training.tif'; 
names{ind}.manual1 = '25_manual1.gif';
names{ind}.aperture = '25_training_mask.gif';

ind = ind+1;
names{ind}.original = '26_training.tif'; 
names{ind}.manual1 = '26_manual1.gif';
names{ind}.aperture = '26_training_mask.gif';

ind = ind+1;
names{ind}.original = '27_training.tif'; 
names{ind}.manual1 = '27_manual1.gif';
names{ind}.aperture = '27_training_mask.gif';

ind = ind+1;
names{ind}.original = '28_training.tif'; 
names{ind}.manual1 = '28_manual1.gif';
names{ind}.aperture = '28_training_mask.gif';

ind = ind+1;
names{ind}.original = '29_training.tif'; 
names{ind}.manual1 = '29_manual1.gif';
names{ind}.aperture = '29_training_mask.gif';

ind = ind+1;
names{ind}.original = '30_training.tif'; 
names{ind}.manual1 = '30_manual1.gif';
names{ind}.aperture = '30_training_mask.gif';

ind = ind+1;
names{ind}.original = '31_training.tif'; 
names{ind}.manual1 = '31_manual1.gif';
names{ind}.aperture = '31_training_mask.gif';

ind = ind+1;
names{ind}.original = '32_training.tif'; 
names{ind}.manual1 = '32_manual1.gif';
names{ind}.aperture = '32_training_mask.gif';

ind = ind+1;
names{ind}.original = '33_training.tif'; 
names{ind}.manual1 = '33_manual1.gif';
names{ind}.aperture = '33_training_mask.gif';

ind = ind+1;
names{ind}.original = '34_training.tif'; 
names{ind}.manual1 = '34_manual1.gif';
names{ind}.aperture = '34_training_mask.gif';

ind = ind+1;
names{ind}.original = '35_training.tif'; 
names{ind}.manual1 = '35_manual1.gif';
names{ind}.aperture = '35_training_mask.gif';

ind = ind+1;
names{ind}.original = '36_training.tif'; 
names{ind}.manual1 = '36_manual1.gif';
names{ind}.aperture = '36_training_mask.gif';

ind = ind+1;
names{ind}.original = '37_training.tif'; 
names{ind}.manual1 = '37_manual1.gif';
names{ind}.aperture = '37_training_mask.gif';

ind = ind+1;
names{ind}.original = '38_training.tif'; 
names{ind}.manual1 = '38_manual1.gif';
names{ind}.aperture = '38_training_mask.gif';

ind = ind+1;
names{ind}.original = '39_training.tif'; 
names{ind}.manual1 = '39_manual1.gif';
names{ind}.aperture = '39_training_mask.gif';

ind = ind+1;
names{ind}.original = '40_training.tif'; 
names{ind}.manual1 = '40_manual1.gif';
names{ind}.aperture = '40_training_mask.gif';
