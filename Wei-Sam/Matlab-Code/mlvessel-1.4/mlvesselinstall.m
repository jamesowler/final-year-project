function mlvesselinstall()

olddir = pwd;

% Finds installation directory and saves it in installdir.m.
disp('Configuring installation directory');

instscript = which('mlvesselinstall');
[instdir,name,ext] = fileparts(instscript);

% Open file for writing.
fid = fopen([instdir filesep 'src' filesep 'tests' filesep ...
             'installdir.m'], 'w');

fprintf(fid, 'function dir = installdir()\n\n');
fprintf(fid, '%s', ['dir = ''' instdir ''';']);

fclose(fid);

% Runs mex to compile skel.c
disp('Compiling C skeleton module with MEX interface.');

cd([instdir filesep 'src' filesep 'skel']);

mex skel.c;

% Returns to initial directory.
cd(olddir);

testpath = [instdir filesep 'src' filesep 'tests'];
guipath = [instdir filesep 'src' filesep 'gui'];

% Adds paths so user can run immediately.
addpath(testpath);
addpath(guipath);

% Now writes executable script for unix and shortcut for windows.
if isunix % This has only been tested on Linux.
  scriptname = [instdir filesep 'mlvessel.sh']; 
  guiscriptname = [instdir filesep 'mlvesselgui.sh']; 
  
  fid = fopen(scriptname, 'w');
  guifid = fopen(guiscriptname, 'w');
    
  fprintf(fid, '#!/bin/sh\n\n');
  fprintf(guifid, '#!/bin/sh\n\n');

  fprintf(fid, ['matlab -r "addpath ' testpath '; addpath ' guipath ...
                '"']);
  fprintf(guifid, ['matlab -r "addpath ' testpath '; addpath ' guipath ...
                   '; guimain"']);
  
  fclose(fid);
  fclose(guifid);
  
  % Sets permission for execution
  unix(['chmod u+x ' scriptname]);
  unix(['chmod u+x ' guiscriptname]);
    
  % Advises user to add the tests directory to his path.
  disp(sprintf(['\nPlease check for any errors...\n\n' ...
    'To run the pakage''s scripts, use the mlvessel.sh script created in\n' ...
    'the mlvessel/ directory to run MATLAB. To run the GUI directly, call\n'...
    'the mlvesselgui.sh script. The main test scripts are in the\n' ...
    'src/tests/ directory and the GUI''s scripts are in the src/gui\n' ...
    'directory. You might want to add those directories to your\n'...
    'startup.m script. Alternatively, start MATLAB with mlvessel.sh or\n' ...
    'mlvesselgui.sh. For further information, see the README.txt.\n\n' ...
    'Call function guimain to open the mlvessel GUI now.\n']));
elseif ispc % Has only been tested on windows XP.
  desktopdir = [getenv('userprofile') filesep 'Desktop'];
    
  % Removes newline from end
  desktopdir = deblank(desktopdir);
  
  scriptname = [desktopdir filesep 'mlvessel.bat'];
  guiscriptname = [desktopdir filesep 'mlvesselgui.bat'];
  
  fid = fopen(scriptname, 'w');
  guifid = fopen(guiscriptname, 'w');
    
  fprintf(fid, '%s', ['start matlab /r "addpath(''' testpath '''); addpath (''' guipath ...
                ''')"']);
  fprintf(guifid, '%s', ['start matlab /r "addpath(''' testpath '''); addpath (''' guipath ...
                   '''); guimain"']);
  
  fclose(fid);
  fclose(guifid);
    
  % Advises user to add the tests directory to his path.
  disp(sprintf(['\nPlease check for any errors...\n\n' ...
    'To run the pakage''s scripts, use the mlvessel.bat script created on\n' ...
    'your desktop to run MATLAB. To run the GUI directly, call\n'...
    'the mlvesselgui.bat script. The main test scripts are in the\n' ...
    'src/tests/ directory and the GUI''s scripts are in the src/gui\n' ...
    'directory. You might want to add those directories to your\n'...
    'startup.m script. Alternatively, start MATLAB with mlvessel.bat or\n' ...
    'mlvesselgui.bat. For further information, see the README.txt.\n\n' ...
    'Call function guimain to open the mlvessel GUI now.\n']));
end
