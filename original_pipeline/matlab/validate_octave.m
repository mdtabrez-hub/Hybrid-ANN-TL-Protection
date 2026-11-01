%% validate_octave.m
%  Phase 2 of the Hybrid ANN-Proteus framework, executed under GNU Octave.
%
%  This script performs the SAME arithmetic as validate_generate_c.m for
%  stages 2.1 (reconstruct), 2.2 (cross-validate) and the int8 quantisation
%  study, and additionally emits the C implementation directly.
%
%  WHAT IT IS NOT: it is not MATLAB, and it does not call MATLAB Coder.
%  The `codegen` invocation in validate_generate_c.m is the one step that has
%  no Octave equivalent; the C here is emitted by emit_c.m, a hand-written
%  generator that implements the same explicit Wx+b forward pass. Any claim
%  specific to MATLAB Coder's own output, and any on-target timing, remains
%  outside what this script can establish.
%
%  Octave compatibility notes:
%    * coder.load  -> supplied by the +coder/ shim (semantically LOAD)
%    * readmatrix  -> dlmread(file, ',', 1, 0), skipping the header row
%    * quantizenumeric (Fixed-Point Designer) -> explicit round/clip, which is
%      what Eq. (21) specifies anyway
%
%  Usage:  octave --no-gui --quiet validate_octave.m

more off;
outDir = '../out';
resDir = '../out/phase2';
if exist(resDir, 'dir') ~= 7, mkdir(resDir); end

printf('================================================================\n');
printf('PHASE 2 -- ANN validation and C generation (GNU Octave %s)\n', version);
printf('================================================================\n');

%% ---------------------------------------------------------------- 2.1 load
S = load(fullfile(outDir, 'ann_weights.mat'));
printf('\n[2.1] Reconstruct the network from ann_weights.mat\n');
printf('  W1 %dx%d   W2 %dx%d   W3 %dx%d   W4 %dx%d\n', ...
       size(S.W1), size(S.W2), size(S.W3), size(S.W4));
nParam = numel(S.W1)+numel(S.b1)+numel(S.W2)+numel(S.b2)+ ...
         numel(S.W3)+numel(S.b3)+numel(S.W4)+numel(S.b4);
printf('  trainable parameters: %d\n', nParam);

X_test    = dlmread(fullfile(outDir, 'X_test.csv'),       ',', 1, 0);
y_test    = dlmread(fullfile(outDir, 'y_test.csv'),       ',', 1, 0);
y_pred_py = dlmread(fullfile(outDir, 'y_pred_python.csv'),',', 1, 0);
N = size(X_test, 1);
printf('  test set: %d samples x %d features\n', N, size(X_test, 2));

%% ------------------------------------------------- 2.2 cross-validation
% Explicit forward pass, single precision, identical to ann_predict.m.
W1 = single(S.W1); b1 = single(S.b1(:)');
W2 = single(S.W2); b2 = single(S.b2(:)');
W3 = single(S.W3); b3 = single(S.b3(:)');
W4 = single(S.W4); b4 = single(S.b4(:)');

tic;
y_pred_ml = false(N, 4);
scores    = zeros(N, 4, 'single');
for i = 1:N
    x  = single(X_test(i,:));
    a1 = max(x  * W1 + b1, single(0));
    a2 = max(a1 * W2 + b2, single(0));
    a3 = max(a2 * W3 + b3, single(0));
    z4 = a3 * W4 + b4;
    p  = single(1) ./ (single(1) + exp(-z4));
    scores(i,:)    = p;
    y_pred_ml(i,:) = p > single(0.5);
end
tInfer = toc;

nOut      = numel(y_pred_ml);
nDisagree = sum(abs(double(y_pred_ml(:)) - double(y_pred_py(:))));
D         = nDisagree / nOut * 100;                    % Eq. (20)
accOct    = mean(all(y_pred_ml == y_test, 2)) * 100;
accPy     = mean(all(y_pred_py == y_test, 2)) * 100;

printf('\n[2.2] Cross-validation against the Python predictions\n');
printf('  compared outputs         : %d  (%d samples x 4)\n', nOut, N);
printf('  disagreeing outputs      : %d\n', nDisagree);
printf('  discrepancy D            : %.6f %%   (criterion: < 0.1 %%)\n', D);
printf('  exact-match acc (Octave) : %.4f %%\n', accOct);
printf('  exact-match acc (Python) : %.4f %%\n', accPy);
printf('  mean inference time      : %.4f ms/sample (host, NOT the MCU)\n', tInfer/N*1e3);
if D < 0.1
    verdict = 'PASS';
else
    verdict = 'FAIL';
end
printf('  RESULT: %s\n', verdict);

fid = fopen(fullfile(resDir, 'validation_report.txt'), 'w');
fprintf(fid, 'ANN forward-pass cross-validation\n');
fprintf(fid, 'engine                : GNU Octave %s (NOT MATLAB; no MATLAB Coder)\n', version);
fprintf(fid, 'generated             : %s\n\n', datestr(now));
fprintf(fid, 'test samples          : %d\n', N);
fprintf(fid, 'compared outputs      : %d\n', nOut);
fprintf(fid, 'disagreeing outputs   : %d\n', nDisagree);
fprintf(fid, 'discrepancy D [%%]     : %.8f\n', D);
fprintf(fid, 'criterion             : D < 0.1 %%\n');
fprintf(fid, 'verdict               : %s\n', verdict);
fprintf(fid, 'exact-match acc Octave: %.6f %%\n', accOct);
fprintf(fid, 'exact-match acc Python: %.6f %%\n', accPy);
fprintf(fid, 'host inference [ms]   : %.6f\n', tInfer/N*1e3);
fclose(fid);

csvwrite(fullfile(resDir, 'y_pred_octave.csv'), double(y_pred_ml));
csvwrite(fullfile(resDir, 'scores_octave.csv'), double(scores));

%% --------------------------------------- int8 quantisation study, Eq. (21)
printf('\n[2.3] int8 symmetric per-tensor quantisation, Eq. (21)\n');
names  = {'W1','b1','W2','b2','W3','b3','W4','b4'};
scales = zeros(1, numel(names));
Q = struct();
fq = fopen(fullfile(resDir, 'quant_report.txt'), 'w');
fprintf(fq, 'int8 symmetric per-tensor quantisation (Eq. 21)\n');
fprintf(fq, 'engine: GNU Octave %s\n\n', version);
for k = 1:numel(names)
    Wk = S.(names{k});
    s  = max(abs(Wk(:))) / 127;
    q  = round(Wk / s);
    q(q >  127) =  127;
    q(q < -128) = -128;
    err = max(abs(Wk(:) - s * q(:)));
    rel = err / max(abs(Wk(:))) * 100;
    scales(k) = s;
    Q.(names{k}) = q;
    printf('  %-3s n=%5d  scale=%.6e  max|err|=%.6e  (%.4f %% of max|W|)\n', ...
           names{k}, numel(Wk), s, err, rel);
    fprintf(fq, '%-3s n=%d scale=%.10e max_abs_err=%.10e rel_pct=%.6f\n', ...
            names{k}, numel(Wk), s, err, rel);
end

% Accuracy actually retained after quantisation -- measured, not assumed.
qW1 = single(Q.W1) * scales(1);  qb1 = single(Q.b1(:)') * scales(2);
qW2 = single(Q.W2) * scales(3);  qb2 = single(Q.b2(:)') * scales(4);
qW3 = single(Q.W3) * scales(5);  qb3 = single(Q.b3(:)') * scales(6);
qW4 = single(Q.W4) * scales(7);  qb4 = single(Q.b4(:)') * scales(8);
y_pred_q = false(N, 4);
for i = 1:N
    x  = single(X_test(i,:));
    a1 = max(x  * qW1 + qb1, single(0));
    a2 = max(a1 * qW2 + qb2, single(0));
    a3 = max(a2 * qW3 + qb3, single(0));
    p  = single(1) ./ (single(1) + exp(-(a3 * qW4 + qb4)));
    y_pred_q(i,:) = p > single(0.5);
end
accQ  = mean(all(y_pred_q == y_test, 2)) * 100;
Dq    = sum(abs(double(y_pred_q(:)) - double(y_pred_py(:)))) / nOut * 100;
printf('\n  exact-match accuracy, float32 : %.4f %%\n', accOct);
printf('  exact-match accuracy, int8    : %.4f %%\n', accQ);
printf('  accuracy change               : %+.4f percentage points\n', accQ - accOct);
printf('  discrepancy vs Python, int8   : %.6f %%\n', Dq);
fprintf(fq, '\nfloat32_exact_match_pct=%.6f\n', accOct);
fprintf(fq, 'int8_exact_match_pct=%.6f\n', accQ);
fprintf(fq, 'delta_pp=%.6f\n', accQ - accOct);
fprintf(fq, 'int8_discrepancy_vs_python_pct=%.6f\n', Dq);
fprintf(fq, 'float32_bytes=%d\nint8_bytes=%d\natmega328p_sram_bytes=2048\n', ...
        nParam*4, nParam);
fclose(fq);
csvwrite(fullfile(resDir, 'y_pred_octave_int8.csv'), double(y_pred_q));

%% --------------------------------------------------- emit C (not Coder)
printf('\n[2.3b] Emitting C (hand-written generator, NOT MATLAB Coder)\n');
emit_c(S, Q, scales, resDir);
printf('  wrote ann_predict.c / ann_predict.h / ann_predict_data.c\n');

%% ------------------------------------------------------------- summary
sfid = fopen(fullfile(resDir, 'phase2_summary.csv'), 'w');
fprintf(sfid, 'metric,value,unit,note\n');
fprintf(sfid, 'engine,Octave %s,,not MATLAB; no Coder\n', version);
fprintf(sfid, 'test_samples,%d,samples,\n', N);
fprintf(sfid, 'compared_outputs,%d,values,\n', nOut);
fprintf(sfid, 'disagreeing_outputs,%d,values,\n', nDisagree);
fprintf(sfid, 'discrepancy_D,%.8f,%%,Eq. (20); criterion < 0.1\n', D);
fprintf(sfid, 'verdict,%s,,\n', verdict);
fprintf(sfid, 'exact_match_float32,%.6f,%%,\n', accOct);
fprintf(sfid, 'exact_match_int8,%.6f,%%,Eq. (21)\n', accQ);
fprintf(sfid, 'int8_delta,%.6f,pp,\n', accQ - accOct);
fprintf(sfid, 'discrepancy_D_int8,%.8f,%%,\n', Dq);
fprintf(sfid, 'host_inference_time,%.6f,ms/sample,host CPU not MCU\n', tInfer/N*1e3);
fprintf(sfid, 'trainable_parameters,%d,weights,\n', nParam);
fclose(sfid);
printf('\nWrote -> %s\n', resDir);
printf('================================================================\n');
