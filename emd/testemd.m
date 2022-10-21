function [f, fval] = testemd(f1, f2, w1, w2)

% Earth Mover's Distance
[f, fval] = emd(f1, f2, w1, w2, @gdf);

end