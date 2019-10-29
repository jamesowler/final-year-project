ABOUT
=====

    This is mlvessel, version 1.4. mlvessel is a Matlab package for
retinal vessel segmentation. The package can be used in two different
manners: through the graphical user interface or through script
invocation. Use through script invocation relies on input files and
produces both file and html output. You may also download a
stand-alone executable corresponding to the graphical user interface
at our website mentioned below, though some functionalities are only
available through script invocation.

      For more information on the methods, people involved, and also
the package's repository, see:

  http://retinal.sourceforge.net

This package was developed as part of the MSc project of
J. V. B. Soares. People who have contributed in the coding of this
package are:

- J. V. B. Soares <joao@vision.ime.usp.br>
- J. J. G. Leandro <jleandro@vision.ime.usp.br>
- R. M. Cesar Jr. <cesar@vision.ime.usp.br>
- E. L. N Tozette <emr@linux.ime.usp.br>

INSTALLING:
===========

1. Unpack the package (mlvessel.zip or mlvessel.tar.gz), using unzip
   or gnuzip and tar. Leave the unzipped files in the directory you
   wish to install the package. If you change the directory, you
   should re-install the package.

2. Run the script mlvesselinstall.m from within Matlab.

3. The main test scripts are in the src/tests/ directory and the GUI's
   scripts are in the src/gui directory. You might want to add those
   directories to your startup.m script. Alternatively, start MATLAB
   with mlvessel.bat (mlvessel.sh) or mlvesselgui.bat
   (mlvesselgui.sh), which are created with the mlveselinstall.m
   script.

IMPORTANT:

- Needs basic Matlab toolboxes (as of version 1.1 no longer needs the
  mathematical morphology toolbox).

- This has only been tested under Linux and Windows XP with 
  Matlab 6.0, 6.5, and 7.12.

- Needs a lot of disk space and memory.

RUNNING:
========

    For GUI use, run the script guimain.m (located in src/gui).
Some features (as the stats module) are only available through
script invocation, though.

    For script invocation, the main tests, which should serve
as examples of use, are:

>> testwindow(someconfig)
   (Tests for training with part of the image being segmented)

>> testmixed(someconfig)
   (Tests for separate train and test sets)

>> testleaveoneout(someconfig)
   (Leave-one-out tests)

"someconfig" describes the configurations, including image file names,
the classifier parameters, features parameters, etc. Examples are in
"stareconfig.m" and "driveconfig.m".

    The images for tests are not included, so you might want to
download the images from the DRIVE and STARE databases for testing.

    The tests' outputs are saved as images, .mat files, and also html
pages.

MODULES:
========

    For examples of use of the modules, see the functions:
"testwindow.m", "testmixed.m" and "testleaveoneout.m". The directories
containing each modules script are in the "src/" directory.

- "gui": contains the graphical user interface as of version 1.1. The
  main window, which provides access to all others is guimain.m.  

- "ftrs": generates and manipulates features. Includes generation of
  the aperture mask and creation of training sets of pixel
  samples. The "createfeatures" functions create a file with the raw
  features of the image provided. The "createlabelled" functions
  receive the features and pixel labels from manual segmentations,
  saving them all together. Finally, the function "createprocessed"
  normalizes the labelled data, preparing it to be fed to a
  classifier. For an example of use, see "test_window.m",
  "test_mixed.m".

- "gmm": The gaussian mixture model functions. "gmmcreatemodel"
  receives a processed file and creates and saves the gaussians
  describing the likelihoods. "gmmclassify" receives the gaussians
  from "gmmcreatemodel" and a raw feature file and produces the class
  for each pixel.

- "html": contains functions for saving the html pages and images.

- "lmse": functions for the linear minumum squered error
  classifier. Creation and application of the classifier is similar to
  the "gmm" module.

- "knn": a knn classifier.

- "skel": contains a function for creating the skeleton of a
  segmentation. It's written in C with a Matlab interface.

- "stats": generates stats for results and puts them on a html page.

CHANGES
=======

Version 1.4 (2012-09-08)

- Fixed an issue with Matlab 7.12, which raised an error when
  variables and functions had the same name
- Removed isgray, isbw, and isrgb, since Matlab says they might become
  obsolete
- Changed stareconfig.m and driveconfig.m so that they are easier to
  use. Changed the image names so that they are directly related to
  the names in the original datasets and also set them to use
  the linear classifier for speed
- Fixed htwimage.m so that it would convert binary images to grayscale
  when resizing for generating better icons

Version 1.3 (2008-07-25)

- Fixed minor bugs
- Added compatibility for building a stand-alone with mcc

Version 1.2 (08/07/2006)

- Fixed minor bugs.
