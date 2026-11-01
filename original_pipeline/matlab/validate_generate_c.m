%% validate_generate_c.m
%  Phase 2 of the Hybrid ANN-Proteus framework (Blueprint v1.0).
%
%  Stage 2.1 - reconstruct the ANN in MATLAB from ann_weights.mat
%  Stage 2.2 - cross-validate MATLAB inference against the Python predictions
%  Stage 2.3 - generate C for the ATmega328P target and quantise to int8
%
%  REQUIREMENTS : MATLAB R2022b+, MATLAB Coder. Deep Learning Toolbox is NOT
%                 required, because the forward pass is written explicitly.
%  INPUTS       : ann_weights.mat, X_test.csv, y_test.csv, y_pred_python.csv
%                 (all produced by train_ann.py, in ../out/)
%  OUTPUTS      : validation_report.txt, codegen/ directory, quant_report.txt
%
%  STATUS: PENDING EXTERNAL EXECUTION. This script has not been run in the
%  present study; no MATLAB result is reported in the manuscript until it is.

clear; clc;
outDir = fullfile('..','out');
assert(isfile(fullfile(outDir,'ann_weights.mat')), ...
    'ann_weights.mat not found - run train_ann.py first.');

%% ---------------------------------------------------------------- 2.1 load
S = load(fullfile(outDir,'ann_weights.mat'));
fprintf('Loaded weights:\n');
fprintf('  W1 %dx%d   W2 %dx%d   W3 %dx%d   W4 %dx%d\n', ...
    size(S.W1), size(S.W2), size(S.W3), size(S.W4));
nParam = numel(S.W1)+numel(S.b1)+numel(S.W2)+numel(S.b2)+ ...
         numel(S.W3)+numel(S.b3)+numel(S.W4)+numel(S.b4);
fprintf('  total trainable parameters: %d\n', nParam);

X_test      = readmatrix(fullfile(outDir,'X_test.csv'),      'NumHeaderLines',1);
y_test      = readmatrix(fullfile(outDir,'y_test.csv'),      'NumHeaderLines',1);
y_pred_py   = readmatrix(fullfile(outDir,'y_pred_python.csv'),'NumHeaderLines',1);
N = size(X_test,1);
fprintf('  test set: %d samples x %d features\n', N, size(X_test,2));

%% -------------------------------------------------- 2.2 cross-validation
y_pred_ml = false(N,4);
t0 = tic;
for i = 1:N
    y_pred_ml(i,:) = ann_predict(X_test(i,:));
end
tInfer = toc(t0);

nOut       = numel(y_pred_ml);
nDisagree  = sum(abs(double(y_pred_ml(:)) - double(y_pred_py(:))));
D          = nDisagree / nOut * 100;                % Eq. (16)
accMATLAB  = mean(all(y_pred_ml == y_test, 2)) * 100;
accPython  = mean(all(y_pred_py == y_test, 2)) * 100;

fprintf('\n--- Stage 2.2 cross-validation ---\n');
fprintf('  disagreeing outputs      : %d / %d\n', nDisagree, nOut);
fprintf('  discrepancy D            : %.4f %%   (acceptance: < 0.1 %%)\n', D);
fprintf('  exact-match acc (MATLAB) : %.4f %%\n', accMATLAB);
fprintf('  exact-match acc (Python) : %.4f %%\n', accPython);
fprintf('  mean inference time      : %.4f ms/sample (host, not MCU)\n', ...
        tInfer/N*1e3);
if D < 0.1
    fprintf('  RESULT: PASS\n');
else
    fprintf('  RESULT: FAIL - investigate dtype, weight order or scaler\n');
end

fid = fopen('validation_report.txt','w');
fprintf(fid,'MATLAB-Python cross-validation\ngenerated %s\n\n', datestr(now));
fprintf(fid,'test samples          : %d\n', N);
fprintf(fid,'disagreeing outputs   : %d / %d\n', nDisagree, nOut);
fprintf(fid,'discrepancy D [%%]     : %.6f\n', D);
fprintf(fid,'acceptance criterion  : D < 0.1 %%\n');
fprintf(fid,'verdict               : %s\n', string(D<0.1).*"PASS"+string(D>=0.1).*"FAIL");
fclose(fid);

%% --------------------------------------------- 2.3 C code generation
cfg                = coder.config('lib');
cfg.TargetLang     = 'C';
cfg.GenerateReport = true;
cfg.GenCodeOnly    = true;
cfg.HardwareImplementation.ProdHWDeviceType = 'Atmel->AVR';
codegen -config cfg ann_predict -args {zeros(1,24,'double')} -d codegen
fprintf('\nGenerated C -> codegen/  (ann_predict.c/.h, ann_predict_data.c)\n');

%% --------------------------------------------- int8 quantisation study
% Per-tensor symmetric quantisation:  s = max|W| / 127,  q = round(W/s)
names = {'W1','b1','W2','b2','W3','b3','W4','b4'};
fq = fopen('quant_report.txt','w');
fprintf(fq,'int8 symmetric per-tensor quantisation\n\n');
fprintf('\n--- int8 quantisation ---\n');
for k = 1:numel(names)
    Wk = S.(names{k});
    s  = max(abs(Wk(:)))/127;
    q  = round(Wk/s);
    q(q> 127) =  127;  q(q<-128) = -128;
    err = max(abs(Wk(:) - s*q(:)));
    fprintf('  %-3s  scale=%.6e  max abs error=%.6e\n', names{k}, s, err);
    fprintf(fq,'%-3s scale=%.8e max_abs_error=%.8e n=%d\n', ...
            names{k}, s, err, numel(Wk));
end
bytes8  = nParam;            % 1 byte per weight
bytes32 = nParam*4;
fprintf('\n  float32 weights : %6d B (%.2f KiB)\n', bytes32, bytes32/1024);
fprintf('  int8    weights : %6d B (%.2f KiB)\n', bytes8,  bytes8/1024);
fprintf('  ATmega328P SRAM : 2048 B  -> weights MUST live in PROGMEM (32 KiB flash)\n');
fprintf(fq,'\nfloat32_bytes=%d\nint8_bytes=%d\natmega328p_sram_bytes=2048\n', ...
        bytes32, bytes8);
fclose(fq);

fprintf('\nNEXT: measure on-target execution time and RAM in Proteus before\n');
fprintf('reporting any trip-latency figure.\n');
