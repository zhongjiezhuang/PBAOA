function [ D ] = load_data( dataname )
%LOAD_DATA    
if strcmp(dataname,'arrhythmia') % 11
    load dataset/arrhythmia.mat
elseif strcmp(dataname,'gastroenterology') % 12
    load dataset/gastroenterology.mat
elseif strcmp(dataname,'LSVT_voice_rehabilitation') % 13
    load dataset/LSVT_voice_rehabilitation.mat
elseif strcmp(dataname,'PersonGait') % 14
    load dataset/PersonGait.mat
elseif strcmp(dataname,'SCADI') % 15
    load dataset/SCADI.mat
elseif strcmp(dataname,'Urban_land_cover') % 16
    load dataset/Urban_land_cover.mat
elseif strcmp(dataname,'ORL') % 17
    load dataset/ORL.mat
elseif strcmp(dataname,'warpAR10P') % 18
    load dataset/warpAR10P.mat
elseif strcmp(dataname,'warpPIE10P') % 19
    load dataset/warpPIE10P.mat
elseif strcmp(dataname,'Yale') % 20
    load dataset/Yale.mat
elseif strcmp(dataname,'CNAE_9') % CNAE_9
    load dataset/CNAE9.mat
elseif strcmp(dataname,'COIL20') % COIL20
    load dataset/COIL20.mat
elseif strcmp(dataname,'lung') % lung
    load dataset/lung.mat
elseif strcmp(dataname,'TOX_171') % TOX_171
    load dataset/TOX_171.mat        
end
D = data;

end

