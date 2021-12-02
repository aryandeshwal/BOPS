% MIT License
%
% Copyright (c) 2017 Jose Bravo Ferreira
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

function [BB_col] = build_BB(n,BLKSZ)
    BB_col = cell(1,BLKSZ);
    d = BLKSZ*n+1;
    for j=1:BLKSZ
        BB = spalloc(n,d^2,2*n);
        for iii=1:n
            temp = zeros(d); temp(iii+(j-1)*n,d) = 1/2; temp(d,iii+(j-1)*n) = 1/2;
            BB(iii,:) = temp(:)';
        end
        BB_col{j} = BB;
    end
end
