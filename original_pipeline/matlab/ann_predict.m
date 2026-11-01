function fault_flags = ann_predict(features) %#codegen
%ANN_PREDICT  Forward pass of the trained 24-64-32-16-4 fault classifier.
%
%   FAULT_FLAGS = ANN_PREDICT(FEATURES) takes a 1x24 row vector of
%   ALREADY-NORMALISED features and returns a 1x4 logical row vector of fault
%   flags in the order [G C B A].
%
%   The forward pass is written out as explicit matrix operations rather than
%   calling PREDICT on a dlnetwork, because MATLAB Coder handles Wx+b
%   efficiently and does not support all dlnetwork paths for embedded targets
%   (Blueprint v1.0, "Practical Tips and Common Pitfalls").
%
%   Weights are loaded once from ann_weights.mat, which is produced by
%   train_ann.py.  coder.load makes the constant available at code-generation
%   time so that the generated C embeds the weights as static arrays.
%
%   PENDING EXTERNAL EXECUTION: this function has NOT been run in the present
%   study.  MATLAB R2022b+ is required.  Run validate_generate_c.m to execute
%   it, measure the Python/MATLAB discrepancy and generate C.

%#codegen
persistent W1 b1 W2 b2 W3 b3 W4 b4
if isempty(W1)
    S  = coder.load('ann_weights.mat');
    W1 = single(S.W1);  b1 = single(S.b1(:)');
    W2 = single(S.W2);  b2 = single(S.b2(:)');
    W3 = single(S.W3);  b3 = single(S.b3(:)');
    W4 = single(S.W4);  b4 = single(S.b4(:)');
end

x  = single(features(:)');            % 1x24

% Keras Dense stores weights as [n_in x n_out], so x*W is the correct order.
a1 = max(x  * W1 + b1, single(0));    % 1x64, ReLU
a2 = max(a1 * W2 + b2, single(0));    % 1x32, ReLU
a3 = max(a2 * W3 + b3, single(0));    % 1x16, ReLU
z4 = a3 * W4 + b4;                    % 1x4,  logits
p  = single(1) ./ (single(1) + exp(-z4));   % sigmoid

fault_flags = p > single(0.5);

% NOTE ON DROPOUT: dropout layers are active only during training. At
% inference Keras applies identity, so they are correctly absent here. A
% MATLAB reconstruction that inserts dropoutLayer objects and calls predict
% will still match, but the explicit form above is the code-generation path.
end
