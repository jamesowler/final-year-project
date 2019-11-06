% normal
example_dir = 'C:\Users\James\Projects\final-year-project\Wei-Sam\data\clahe\drive_n4_plus_clahe_images';
target_dir = 'C:\Users\James\Projects\final-year-project\Wei-Sam\Matlab-Code\mlvessel-1.4';
results_dir = 'C:\Users\James\Projects\final-year-project\Wei-Sam\results';

classifier_results_dir = "C:\Users\James\Projects\final-year-project\Wei-Sam\Matlab-Code\mlvessel-1.4\results\mixed_drive_n4_plus_clahe_gmm";

for i = 0:4
    folder_name = string(fullfile(example_dir));
    dir_to_copy = example_dir + "_" + string(i);
    file_parts = regexp(example_dir,filesep,'split');
    disp(string(file_parts{1, 9}))
    disp(target_dir + "\" + string(file_parts{1, 9}))
    
    to_copy_into = target_dir + "\" + string(file_parts{1, 9});
    
    % copy data directory 
    copyfile(string(dir_to_copy), string(to_copy_into))
    
    % train classifier
    testmixed(drive_clahe_config)
    
    % copy over results
    copyfile(classifier_results_dir, results_dir + "\" + "mixed_drive_n4_plus_clahe_gmm_" + string(i))    
    
end