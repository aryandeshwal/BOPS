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

function [varlist] = find_variables(B,k)
    B = B+B';
    % connect graph (if it's not already connected)
    B = connect_graph(B);
    % start with edges
    varlist = find_pairs(B);
    % merge variables until they are of size k
    for i=1:k-2
        varlist = merge_variables(varlist);
    end
end

function [B] = connect_graph(B)
    G = graph(B);
    bins = conncomp(G);
    % if there is more than one connected component
    bin_order = randperm(max(bins));    
    if sum(bins>1) > 0
        for i=1:max(bins)-1
            % choose random node from random bin i
            pop = find(bins==bin_order(i));
            i1 = pop(randsample(length(pop),1));
            % choose random node from random bin i+1
            pop = find(bins==bin_order(i+1));
            i2 = pop(randsample(length(pop),1));
            % connect the two bins
            B(i1,i2) = 1;
            B(i2,i1) = 1;
        end
    end
end

function [pairs] = find_pairs(B)
    B = B+B';
    [i,j,~] = find(triu(B,1));
    pairs = [i, j];
end

function [varlist_n] = merge_variables(varlist)
    n_vars = size(varlist,1);
    varlist_n = [];
    for i=1:n_vars-1
        cur_var = varlist(i,:);
        % find number of common nodes between current variable and the rest
        common = -ones(1,n_vars);
        for j=i+1:n_vars
            common(j) = sum(ismember(cur_var,varlist(j,:)));
        end
        % sort by number of common nodes
        [~,idx] = sort(common,'descend');
        % merge variable with its closest neighbour
        varlist_n = [varlist_n; merge_variables_aux(cur_var,varlist(idx(1),:))];
    end
    varlist_n = unique(varlist_n,'rows');
end

function [var_n] = merge_variables_aux(var1,var2)
    k = length(var1);
    % total pool of nodes
    all_nodes = union(var1,var2);
    var_n = nchoosek(all_nodes,k+1);
end
