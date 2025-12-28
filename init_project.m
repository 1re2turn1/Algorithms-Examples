% init_project.m
% call this function first when you open the project
function init_project()
    project_root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(project_root, 'FI Fisher Information')));
    % ... 

    cd(project_root);
    
    fprintf('Project initialized successfully!\n');
    fprintf('Current Path: %s\n', pwd);
end