function S = load(varargin)
%CODER.LOAD  Octave/MATLAB-runtime shim for coder.load.
%   In MATLAB Coder, coder.load() makes a MAT-file a compile-time constant.
%   Outside code generation it is semantically identical to LOAD, so this shim
%   lets ann_predict.m execute unmodified under GNU Octave, which has no
%   MATLAB Coder. It changes no arithmetic.
  S = load(varargin{:});
end
