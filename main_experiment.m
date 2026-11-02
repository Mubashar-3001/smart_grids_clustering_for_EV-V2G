function R = main_experiment(caseName, nRep, doSweep, feederName)
%MAIN_EXPERIMENT  Reproduce the paper's results in base MATLAB (no toolboxes).
%
%   R = main_experiment('moderate', 10, false, 'ieee33')    % 33-bus, moderate DER
%   R = main_experiment('der_rich', 10, false, 'ieee69')    % 69-bus, DER-rich
%   R = main_experiment('der_rich', 10, false, 'ieee123')   % 123-bus, DER-rich
%   R = main_experiment('der_rich', 3,  true,  'ieee123')   % + group-count sweep
%
% Line-by-line mirror of ../python/run_experiment.py: same feeders (69- and
% 123-bus data read from ../data, built by prepare_feeders.py), same real
% Ausgrid data, same EV/V2G model, same WOA, same schemes and the same
% assumptions [A1]-[A7]. MATLAB and NumPy random streams differ, so
% compare a run with the paper's mean +/- std over replications, not digit
% by digit. The gradability certificate uses the least-TMS recursion (exact
% for radial feeders, where every relay has one backup; it gives the same
% verdicts and optima as the LP in the Python code), so no toolbox is needed.
%
% Written for MATLAB R2016b+ (implicit expansion); tested in GNU Octave 8.4.

if nargin < 1 || isempty(caseName), caseName = 'moderate'; end
if nargin < 2 || isempty(nRep), nRep = 10; end
if nargin < 3 || isempty(doSweep), doSweep = false; end
if nargin < 4 || isempty(feederName), feederName = 'ieee33'; end
M = setup_model(caseName, feederName);
fprintf('[%s | %s] peak=%.0f kW, %d relays, %d pairs, hubs=%s, rooftop PV=%.0f%% of bus peak\n', ...
    feederName, caseName, M.PEAK_KW, M.NREL, numel(M.PI), mat2str(M.HUB_BUSES), 100*M.ROOF);
names = {'conventional','robust_K1','validity_only','proposed_state','proposed_knn', ...
         'proposed_knn_env','scalar_11','scalar_K'};
reps = cell(nRep, 1);
for r = 0:nRep-1
    t0 = tic;
    res = run_rep(M, r, 480, 480, doSweep);
    reps{r+1} = res;
    fprintf('rep %d: K=%d (%s; validity-best K=%d, %d ungradable) |', r, res.K, res.method, ...
        res.K_validity, res.ungradable_validity);
    for k = 1:numel(names), fprintf(' %s=%d', names{k}, res.(names{k}).total); end
    fprintf(' | acc=%.3f | %.0f s\n', res.proposed_knn.acc, toc(t0));
end
R = summarise(reps, names, [feederName '_' caseName], doSweep);
end

% =====================================================================
% Model
% =====================================================================
function M = setup_model(caseName, feederName)
here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, '..', 'data');
M.SBASE_MVA = 10; M.SBASE_KW = 1e4; M.ZS = 0.20i;
BR33 = [1 2 .0922 .0470; 2 3 .4930 .2511; 3 4 .3660 .1864; 4 5 .3811 .1941;
    5 6 .8190 .7070; 6 7 .1872 .6188; 7 8 .7114 .2351; 8 9 1.0300 .7400;
    9 10 1.0440 .7400; 10 11 .1966 .0650; 11 12 .3744 .1238; 12 13 1.4680 1.1550;
    13 14 .5416 .7129; 14 15 .5910 .5260; 15 16 .7463 .5450; 16 17 1.2890 1.7210;
    17 18 .7320 .5740; 2 19 .1640 .1565; 19 20 1.5042 1.3554; 20 21 .4095 .4784;
    21 22 .7089 .9373; 3 23 .4512 .3083; 23 24 .8980 .7091; 24 25 .8960 .7011;
    6 26 .2030 .1034; 26 27 .2842 .1447; 27 28 1.0590 .9337; 28 29 .8042 .7006;
    29 30 .5075 .2585; 30 31 .9744 .9630; 31 32 .3105 .3619; 32 33 .3410 .5302];
LD33 = [1 0 0; 2 100 60; 3 90 40; 4 120 80; 5 60 30; 6 60 20; 7 200 100;
    8 200 100; 9 60 20; 10 60 20; 11 45 30; 12 60 35; 13 60 35; 14 120 80;
    15 60 10; 16 60 20; 17 60 20; 18 90 40; 19 90 40; 20 90 40; 21 90 40;
    22 90 40; 23 90 50; 24 420 200; 25 420 200; 26 60 25; 27 60 25; 28 60 20;
    29 120 70; 30 200 600; 31 150 70; 32 210 100; 33 60 40];
% relay branches (trunk first, then lateral heads) and hub buses; for the
% 123-bus feeder these are bus indices of feeder123_busmap.csv, i.e. relays
% 149-1, 13-152, 54-57, 60-160, 67-72, 76-86 | 13-18, 60-62, 67-97 and hubs
% 87, 91, 95, 50, 65, 101 in IEEE numbering
switch feederName
    case 'ieee33'
        M.VBASE_KV = 12.66; M.BR = BR33; M.LOADS = LD33; M.NTRUNK = 6;
        edges = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 2 19; 3 23; 6 26];
        M.HUB_BUSES = [9 14 17 21 24 30];
    case 'ieee69'
        M.VBASE_KV = 12.66; M.NTRUNK = 6;
        M.BR = read_csv(fullfile(dataDir, 'feeder69_branches.csv'));
        M.LOADS = read_csv(fullfile(dataDir, 'feeder69_loads.csv'));
        edges = [2 3; 3 4; 5 6; 7 8; 9 10; 11 12; 4 47; 8 51; 9 53];
        M.HUB_BUSES = [12 16 21 49 52 61];
    case 'ieee123'
        M.VBASE_KV = 4.16; M.NTRUNK = 6;
        M.BR = read_csv(fullfile(dataDir, 'feeder123_branches.csv'));
        M.LOADS = read_csv(fullfile(dataDir, 'feeder123_loads.csv'));
        edges = [1 2; 11 13; 31 37; 46 54; 62 69; 75 83; 11 14; 46 56; 62 70];
        M.HUB_BUSES = [91 105 118 72 78 84];
    otherwise
        error('feederName must be ieee33, ieee69 or ieee123');
end
M.ZBASE = M.VBASE_KV^2 / M.SBASE_MVA;
M.IBASE = M.SBASE_MVA*1e6 / (sqrt(3)*M.VBASE_KV*1e3);
M.NBUS = size(M.LOADS, 1); M.NREL = size(edges, 1);
M.PEAK_KW = sum(M.LOADS(:,2));
% pairs: a relay's backup is the nearest relay upstream of its from-bus; the
% pair is checked for a bolted fault at the far end of the primary's branch
parent = zeros(M.NBUS, 1); parent(M.BR(:,2)) = M.BR(:,1);
relAt = zeros(M.NBUS, 1); relAt(edges(:,2)) = 1:M.NREL;
M.PAIRS = zeros(0, 3);
for r = 1:M.NREL
    u = edges(r,1);
    while u > 0
        if relAt(u) > 0, M.PAIRS(end+1,:) = [r, relAt(u), edges(r,2)]; break; end %#ok<AGROW>
        u = parent(u);
    end
end
M.PI = M.PAIRS(:,1)'; M.BI = M.PAIRS(:,2)'; NP = size(M.PAIRS, 1);
Y = zeros(M.NBUS);
for k = 1:size(M.BR, 1)
    f = M.BR(k,1); t = M.BR(k,2); y = M.ZBASE / complex(M.BR(k,3), M.BR(k,4));
    Y(f,f) = Y(f,f) + y; Y(t,t) = Y(t,t) + y; Y(f,t) = Y(f,t) - y; Y(t,f) = Y(t,f) - y;
end
Y(1,1) = Y(1,1) + 1/M.ZS;
M.Zpre = inv(Y);
M.Zflt = cell(NP, 1);
for k = 1:NP
    Yf = Y; fb = M.PAIRS(k,3); Yf(fb,fb) = Yf(fb,fb) + 1e6; M.Zflt{k} = inv(Yf);
end
M.RF = edges(:,1); M.RT = edges(:,2); M.RY = zeros(M.NREL, 1);
for r = 1:M.NREL
    k = find(M.BR(:,1) == edges(r,1) & M.BR(:,2) == edges(r,2));
    M.RY(r) = M.ZBASE / complex(M.BR(k,3), M.BR(k,4));
end
% ---- real data [A1], [A2]
P = read_csv(fullfile(dataDir, 'ausgrid_c12_pv_pu.csv'));
M.DAY_MONTH = P(:,2); M.DAY_WD = P(:,4); M.PV_PU = P(:,5:end); M.NDAYS = size(P,1);
M.LOAD_SHAPE = read_csv(fullfile(dataDir, 'ausgrid_c12_load_shape.csv'));
% ---- hubs [A3]-[A5]; HUB_SCALE identical to Python (numpy seed 2026)
M.HUB_SCALE = [0.743148 1.111931 0.973815 0.896400 0.883934 1.232415];
M.P_EV = 7.2; M.N_EVSE = round(50*M.HUB_SCALE); M.SITE_CAP = 3.75*M.N_EVSE;
M.SESS_WD = 0.8; M.SESS_WE = 0.25; M.V2G_SHARE = 0.30; M.V2G_PROB = 0.35;
M.KF = 2.0; M.PV_ONLINE = 0.02;
M.GMM_WD.w = [0.60 0.25 0.15];
M.GMM_WD.mu = [8.25 8.5 9.0; 12.5 3.5 6.0; 18.0 4.0 7.0];
M.GMM_WD.sd = [1.0 1.5 3.0; 2.0 1.5 2.0; 1.5 2.0 2.5];
M.GMM_WE.w = 1; M.GMM_WE.mu = [13.0 4.0 7.0]; M.GMM_WE.sd = [3.5 2.0 2.5];
switch caseName
    case 'moderate', hubPV = 300; M.ROOF = 0.0;
    case 'der_rich', hubPV = 600; M.ROOF = 0.6;
    otherwise, error('caseName must be ''moderate'' or ''der_rich''');
end
M.PV_RATED = hubPV*M.HUB_SCALE;
% ---- protection
M.A_IEC = 13.5; M.CTI = 0.30; M.CTI_DESIGN = 0.31; M.T_MIN = 0.10;
M.TMS_LO = 0.05; M.TMS_HI = 1.10; M.OLF = 1.25; M.PK_MIN = 50;
M.W_CTI = 500; M.W_BLIND = 800; M.T_BLIND = 5; M.N_AGENTS = 40; M.N_ITERS = 400;
end

function A = read_csv(f)
if exist('readmatrix', 'file') == 2
    A = readmatrix(f);
else
    A = dlmread(f, ',', 1, 0);            % GNU Octave / older MATLAB
end
end

% =====================================================================
% Scenarios and currents
% =====================================================================
function S = generate(M, n, seed)
rng(seed);
day = randi(M.NDAYS, n, 1); slot = randi(48, n, 1) - 1;
hour = slot*0.5 + 0.5*rand(n, 1);
weekend = M.DAY_WD(day) >= 5;
s_load = M.LOAD_SHAPE(sub2ind(size(M.LOAD_SHAPE), M.DAY_MONTH(day), slot + 1));
pv = M.PV_PU(sub2ind(size(M.PV_PU), day, slot + 1));
net = zeros(n, 6); infeed = zeros(n, 6); ev = zeros(n, 6); v2g = zeros(n, 6);
for i = 1:n
    if weekend(i), G = M.GMM_WE; rate = M.SESS_WE; else, G = M.GMM_WD; rate = M.SESS_WD; end
    for h = 1:6
        m = poisson_knuth(rate*M.N_EVSE(h)); imp = 0; ex = 0;
        if m > 0
            comp = sample_weights(G.w, m);
            x = G.mu(comp,:) + G.sd(comp,:).*randn(m, 3);
            arr = mod(x(:,1), 24); dur = min(max(x(:,2), 0.25), 14); en = min(max(x(:,3), 1), 40);
            dt = mod(hour(i) - arr, 24);
            conn = dt < dur;
            chg = conn & (dt < min(en/M.P_EV, dur));              % uncontrolled
            cap = rand(m, 1) < M.V2G_SHARE;
            expo = conn & ~chg & cap & (rand(m, 1) < M.V2G_PROB);
            imp = min(M.P_EV*sum(chg), M.SITE_CAP(h));
            ex = min(M.P_EV*sum(expo), M.SITE_CAP(h));
        end
        net(i,h) = imp - ex - M.PV_RATED(h)*pv(i);
        infeed(i,h) = M.KF*(ex + M.PV_RATED(h)*(pv(i) > M.PV_ONLINE));
        ev(i,h) = imp; v2g(i,h) = ex;
    end
end
S = struct('hour', hour, 's_load', s_load, 'pv', pv, 'net', net, ...
           'infeed', infeed, 'ev', ev, 'v2g', v2g, 'weekend', weekend);
end

function k = poisson_knuth(lambda)
L = exp(-lambda); k = 0; p = rand();
while p > L, k = k + 1; p = p*rand(); end
end

function idx = sample_weights(w, m)
c = cumsum(w(:)')/sum(w);
idx = min(sum(rand(m, 1) > c, 2) + 1, numel(w));
end

function [Lmag, Lfwd, prim, back] = currents(M, S)
% Relay load currents: magnitude (pickup sizing) and forward-only
% (encroachment, directional element) [A6]; per pair, primary and backup
% fault currents -- each relay sees its OWN current.
n = numel(S.s_load);
I = zeros(M.NBUS, n); I(1,:) = 1/M.ZS;
for k = 1:size(M.LOADS, 1)
    b = M.LOADS(k,1); p = M.LOADS(k,2); q = M.LOADS(k,3);
    I(b,:) = I(b,:) - (p + 1i*q)/M.SBASE_KW*S.s_load.' + M.ROOF*p*S.pv.'/M.SBASE_KW;
end
for h = 1:6
    b = M.HUB_BUSES(h); I(b,:) = I(b,:) - S.net(:,h).'/M.SBASE_KW;
end
[amps, fwd] = relay_flows(M, M.Zpre*I);
Lmag = amps; Lfwd = amps.*fwd;
J = zeros(M.NBUS, n); J(1,:) = 1/M.ZS;
for k = 1:size(M.LOADS, 1)
    b = M.LOADS(k,1); p = M.LOADS(k,2);
    J(b,:) = J(b,:) + M.KF*M.ROOF*p*(S.pv.' > M.PV_ONLINE)/M.SBASE_KW;
end
for h = 1:6
    b = M.HUB_BUSES(h); J(b,:) = J(b,:) + S.infeed(:,h).'/M.SBASE_KW;
end
NP = numel(M.PI);
prim = zeros(n, NP); back = zeros(n, NP);
for k = 1:NP
    a = relay_flows(M, M.Zflt{k}*J);
    prim(:,k) = a(:, M.PI(k)); back(:,k) = a(:, M.BI(k));
end
end

function [amps, fwd] = relay_flows(M, V)
Ib = (V(M.RF,:) - V(M.RT,:)).*M.RY;
fwd = (real(V(M.RF,:).*conj(Ib)) > 0).';
amps = (abs(Ib)*M.IBASE).';
end

function c = online_capacity(M, S)
c = sum(S.infeed, 2)/M.KF + M.ROOF*M.PEAK_KW*(S.pv > M.PV_ONLINE);
end

function X = cluster_features(M, S)
X = [S.net, M.PEAK_KW*S.s_load, online_capacity(M, S)];
end

function X = measurable_features(M, S)
head = M.PEAK_KW*S.s_load + sum(S.net, 2) - M.ROOF*M.PEAK_KW*S.pv;
X = [head, sum(S.net(:,1:3), 2), sum(S.net(:,4:6), 2), online_capacity(M, S), ...
     sin(2*pi*S.hour/24), cos(2*pi*S.hour/24)];
end

% =====================================================================
% Coordination: IEC very inverse, robust fitness, WOA
% =====================================================================
function t = op_time(tms, I, Ip)
Mx = I./Ip + zeros(size(tms));
t = max(tms.*13.5./(Mx - 1), 0.10);
t(Mx <= 1 + 1e-9) = Inf;
end

function pk = pickups(M, Lmag_rows)
pk = max(M.OLF*max(Lmag_rows, [], 1), M.PK_MIN);
end

function f = fitness_pop(M, T, prim, back, pk)
% Robust (minimax) fitness of P candidate TMS vectors over S scenarios.
P = size(T, 1); S = size(prim, 1); np = numel(M.PI);
tp = op_time(reshape(T(:, M.PI), P, 1, np), reshape(prim, 1, S, np), reshape(pk(M.PI), 1, 1, np));
tb = op_time(reshape(T(:, M.BI), P, 1, np), reshape(back, 1, S, np), reshape(pk(M.BI), 1, 1, np));
bp = ~isfinite(tp); bb = ~isfinite(tb);
d = tb - tp; d(bp | bb) = M.CTI_DESIGN;
tpe = tp; tpe(bp) = M.T_BLIND;
F = sum(tpe + M.W_BLIND*(double(bp) + double(bb)) + M.W_CTI*max(M.CTI_DESIGN - d, 0), 3);
f = max(F, [], 2) + 0.01*mean(F, 2);
end

function [best, fbest, hist] = woa(M, fitfun, seed)
% Whale Optimization Algorithm (Mirjalili & Lewis, 2016), vector A and C.
rng(seed); n = M.N_AGENTS; d = M.NREL; it = M.N_ITERS;
X = M.TMS_LO + (M.TMS_HI - M.TMS_LO)*rand(n, d);
f = fitfun(X); [fbest, i] = min(f); best = X(i,:); hist = zeros(it + 1, 1); hist(1) = fbest;
for t = 0:it-1
    a = 2 - 2*t/it;
    A = 2*a*rand(n, d) - a; C = 2*rand(n, d);
    p = rand(n, 1); l = -1 + 2*rand(n, d);
    Xr = X(randi(n, n, 1), :);
    small = sqrt(sum(A.^2, 2)) < 1;
    enc = best - A.*abs(C.*best - X);
    exr = Xr - A.*abs(C.*Xr - X);
    Xn = abs(best - X).*exp(l).*cos(2*pi*l) + best;          % spiral
    Xn(p < 0.5 & small, :) = enc(p < 0.5 & small, :);
    Xn(p < 0.5 & ~small, :) = exr(p < 0.5 & ~small, :);
    X = min(max(Xn, M.TMS_LO), M.TMS_HI);
    f = fitfun(X); [fm, i] = min(f);
    if fm < fbest, fbest = fm; best = X(i,:); end
    hist(t + 2) = fbest;
end
end

function T = optimise(M, prim, back, pk, seed)
T = woa(M, @(X) fitness_pop(M, X, prim, back, pk), seed);
end

function T = min_tms(M, prim, back, pk)
% Least feasible TMS vector, bottom-up (each relay after the relays it backs
% up); [] if the group cannot be graded (exact gradability certificate).
Mp = prim./pk(M.PI); Mb = back./pk(M.BI); T = [];
if any(Mp(:) <= 1 + 1e-9) || any(Mb(:) <= 1 + 1e-9), return; end
kp = 13.5./(Mp - 1); kb = 13.5./(Mb - 1);
Tm = NaN(1, M.NREL); done = false(1, M.NREL);
while ~all(done)
    for r = find(~done)
        ks = find(M.BI == r);
        if all(done(M.PI(ks)))
            t = M.TMS_LO;
            for k = ks(:)'
                tp = max(kp(:, k)*Tm(M.PI(k)), M.T_MIN);
                t = max(t, max((tp + M.CTI_DESIGN)./kb(:, k)));
            end
            if t > M.TMS_HI + 1e-12, return; end
            Tm(r) = t; done(r) = true;
        end
    end
end
T = Tm;
end

function ok = gradable(M, prim, back, pk)
ok = ~isempty(min_tms(M, prim, back, pk));
end

function E = evaluate(M, Trows, PKrows, Lfwd, prim, back)
tp = op_time(Trows(:, M.PI), prim, PKrows(:, M.PI));
tb = op_time(Trows(:, M.BI), back, PKrows(:, M.BI));
blind = ~isfinite(tp) | ~isfinite(tb);
marg = tb - tp; marg(blind) = NaN;
cti = ~blind & (marg < M.CTI - 1e-9);
enc = Lfwd > PKrows;
per = sum(cti, 2) + sum(blind, 2) + sum(enc, 2);
E.total = sum(per); E.cti = sum(cti(:)); E.blind = sum(blind(:)); E.enc = sum(enc(:));
E.affected = 100*mean(per > 0);
E.t_prim = mean(tp(isfinite(tp))); E.t_back = mean(tb(isfinite(tb)));
E.pk_trunk = mean(mean(PKrows(:, 1:M.NTRUNK)));
E.min_margin = min(marg(~isnan(marg)));
E.severe = sum(marg(:) < 0.2);
E.groups = NaN;
end

% =====================================================================
% Clustering, validity, classifier (all from scratch)
% =====================================================================
function Z = stdz(Xtr, X)
mu = mean(Xtr, 1); s = std(Xtr, 1, 1); s(s == 0) = 1;
Z = (X - mu)./s;
end

function D = sqdist(X, C)
D = max(sum(X.^2, 2) + sum(C.^2, 2)' - 2*X*C', 0);
end

function lab = kmeans_c(X, K, nInit, seed)
rng(seed); n = size(X, 1); bestI = Inf; bestLab = ones(n, 1);
for r = 1:nInit
    C = X(randperm(n, K), :); lab = zeros(n, 1);
    for it = 1:300
        [~, nl] = min(sqdist(X, C), [], 2);
        if isequal(nl, lab), break; end
        lab = nl;
        for c = 1:K
            if any(lab == c), C(c,:) = mean(X(lab == c, :), 1); end
        end
    end
    inertia = sum(min(sqdist(X, C), [], 2));
    if inertia < bestI, bestI = inertia; bestLab = lab; end
end
[~, ~, lab] = unique(bestLab);
end

function L = ward_multi(X, Ks)
% Ward-linkage agglomeration (Lance-Williams), one pass, labels at each K in Ks.
n = size(X, 1); L = cell(max(Ks), 1);
D = Inf(2*n - 1); D0 = sqdist(X, X); D0(1:n+1:end) = Inf; D(1:n, 1:n) = D0;
sz = [ones(n, 1); zeros(n - 1, 1)];
members = cell(2*n - 1, 1); for i = 1:n, members{i} = i; end
active = 1:n; nxt = n + 1;
while numel(active) > min(Ks)
    sub = D(active, active);
    [~, li] = min(sub(:)); [ii, jj] = ind2sub(size(sub), li);
    a = active(ii); b = active(jj);
    rest = active(active ~= a & active ~= b);
    na = sz(a); nb = sz(b); nk = sz(rest);
    dnew = ((na + nk).*D(a, rest)' + (nb + nk).*D(b, rest)' - nk.*D(a, b))./(na + nb + nk);
    c = nxt; nxt = nxt + 1;
    D(c, rest) = dnew'; D(rest, c) = dnew;
    sz(c) = na + nb; members{c} = [members{a}, members{b}];
    active = [rest, c];
    if any(Ks == numel(active))
        lab = zeros(n, 1);
        for q = 1:numel(active), lab(members{active(q)}) = q; end
        L{numel(active)} = lab;
    end
end
end

function s = silhouette_avg(Dm, lab)
K = max(lab); n = numel(lab); Ssum = zeros(n, K); cnt = zeros(1, K);
for c = 1:K, idx = lab == c; cnt(c) = sum(idx); Ssum(:,c) = sum(Dm(:, idx), 2); end
own = sub2ind([n K], (1:n)', lab(:));
a = Ssum(own)./max(cnt(lab)' - 1, 1);
B = Ssum./cnt; B(own) = Inf; b = min(B, [], 2);
si = (b - a)./max(a, b); si(cnt(lab)' == 1) = 0;
s = mean(si);
end

function db = davies_bouldin(X, lab)
K = max(lab); C = zeros(K, size(X, 2)); sc = zeros(K, 1);
for c = 1:K
    Xi = X(lab == c, :); C(c,:) = mean(Xi, 1); sc(c) = mean(sqrt(sum((Xi - C(c,:)).^2, 2)));
end
Rm = zeros(K);
for i = 1:K
    for j = 1:K
        if i ~= j, Rm(i,j) = (sc(i) + sc(j))/norm(C(i,:) - C(j,:)); end
    end
end
db = mean(max(Rm, [], 2));
end

function [rows, best] = validity(Xs)
Ks = 2:8; Dm = sqrt(sqdist(Xs, Xs)); Dm(1:size(Dm,1)+1:end) = 0;
W = ward_multi(Xs, Ks);
rows = struct('method', {}, 'K', {}, 'sil', {}, 'db', {});
for K = Ks
    lk = kmeans_c(Xs, K, 10, 0);
    rows(end+1) = struct('method', 'kmeans', 'K', K, 'sil', silhouette_avg(Dm, lk), 'db', davies_bouldin(Xs, lk)); %#ok<AGROW>
    lw = W{K};
    rows(end+1) = struct('method', 'ward', 'K', K, 'sil', silhouette_avg(Dm, lw), 'db', davies_bouldin(Xs, lw)); %#ok<AGROW>
end
[~, order] = sortrows([-[rows.sil]', [rows.db]']);
best = rows(order(1));
end

function lab = cluster_labels(Xs, method, K)
if strcmp(method, 'kmeans')
    lab = kmeans_c(Xs, K, 10, 0);
else
    W = ward_multi(Xs, K); lab = W{K};
end
end

function a = nearest(C, X)
[~, a] = min(sqdist(X, C), [], 2);
end

function pred = knn_predict(Xtr, ytr, Xte, k)
[~, order] = sort(sqdist(Xte, Xtr), 2);
nn = reshape(ytr(order(:, 1:k)), [], k);
pred = mode(nn, 2);
end

% =====================================================================
% Setting groups and schemes
% =====================================================================
function [T, PK] = group_settings(M, lab, K, Lmag, prim, back, seed)
T = zeros(K, M.NREL); PK = zeros(K, M.NREL);
for c = 1:K
    idx = lab == c;
    PK(c,:) = pickups(M, Lmag(idx, :));
    T(c,:) = optimise(M, prim(idx, :), back(idx, :), PK(c,:), seed + c - 1);
end
end

function [T, PK, btr, bte] = scalar_groups(M, nb, Str, Ste, LMtr, Ptr, Btr, seed)
% Single aggregate-DER index rho = sum|hub net kW|, nb bins, nearest-bin lookup.
rtr = sum(abs(Str.net), 2); lo = min(rtr); hi = max(rtr);
rtr = (rtr - lo)/(hi - lo);
rte = min(max((sum(abs(Ste.net), 2) - lo)/(hi - lo), 0), 1);
cen = linspace(0, 1, nb);
[~, btr] = min(abs(rtr - cen), [], 2); [~, bte] = min(abs(rte - cen), [], 2);
T = zeros(nb, M.NREL); PK = zeros(nb, M.NREL);
for j = 1:nb
    idx = find(btr == j);
    if numel(idx) < 10, [~, o] = sort(abs(rtr - cen(j))); idx = o(1:10); end
    PK(j,:) = pickups(M, LMtr(idx, :));
    T(j,:) = optimise(M, Ptr(idx, :), Btr(idx, :), PK(j,:), seed + j - 1);
end
end

function res = run_rep(M, r, nTr, nTe, doSweep)
Str = generate(M, nTr, 1000 + r); Ste = generate(M, nTe, 5000 + r);
[LMtr, LFtr, Ptr, Btr] = currents(M, Str);
[LMte, LFte, Pte, Bte] = currents(M, Ste);
seed = 10000 + 100*r;
ev = @(T, PK) evaluate(M, T, PK, LFte, Pte, Bte);
% (a) conventional: one group designed at the peak-load, no-DER base case
S0 = struct('s_load', 1, 'pv', 0, 'net', zeros(1, 6), 'infeed', zeros(1, 6));
[LM0, ~, P0, B0] = currents(M, S0);
pk0 = pickups(M, LM0); T0 = optimise(M, P0, B0, pk0, seed + 1);
res.conventional = ev(repmat(T0, nTe, 1), repmat(pk0, nTe, 1)); res.conventional.groups = 1;
% (b) one robust group over all training scenarios (K = 1)
pk1 = pickups(M, LMtr); T1 = optimise(M, Ptr, Btr, pk1, seed + 2);
res.robust_K1 = ev(repmat(T1, nTe, 1), repmat(pk1, nTe, 1)); res.robust_K1.groups = 1;
% (c) proposed: optimized setting-group finding -- candidate partitions in
%     order of decreasing silhouette (ties: Davies-Bouldin), first one whose
%     every group is gradable; K = 1 if none. Validity-only = silhouette-best.
Xtr = cluster_features(M, Str); Xte = cluster_features(M, Ste);
Xs_tr = stdz(Xtr, Xtr); Xs_te = stdz(Xtr, Xte);
[rows, best] = validity(Xs_tr);
[~, order] = sortrows([-[rows.sil]', [rows.db]']);
labV = cluster_labels(Xs_tr, best.method, best.K);
ungV = 0;
for c = 1:best.K
    idx = labV == c; ungV = ungV + ~gradable(M, Ptr(idx,:), Btr(idx,:), pickups(M, LMtr(idx,:)));
end
K = 1; meth = 'none'; lab = ones(nTr, 1);
for q = order(:)'
    lq = cluster_labels(Xs_tr, rows(q).method, rows(q).K); ok = true;
    for c = 1:rows(q).K
        idx = lq == c;
        if ~gradable(M, Ptr(idx,:), Btr(idx,:), pickups(M, LMtr(idx,:))), ok = false; break; end
    end
    if ok, K = rows(q).K; meth = rows(q).method; lab = lq; break; end
end
if K > 1
    C = zeros(K, size(Xs_tr, 2)); for c = 1:K, C(c,:) = mean(Xs_tr(lab == c, :), 1); end
    [Tk, PKk] = group_settings(M, lab, K, LMtr, Ptr, Btr, seed + 10);
    aS = nearest(C, Xs_te);
else
    Tk = T1; PKk = pk1; aS = ones(nTe, 1);
end
if K == best.K && strcmp(meth, best.method)
    res.validity_only = ev(Tk(aS,:), PKk(aS,:));
else
    CV = zeros(best.K, size(Xs_tr, 2)); for c = 1:best.K, CV(c,:) = mean(Xs_tr(labV == c, :), 1); end
    [Tv, PKv] = group_settings(M, labV, best.K, LMtr, Ptr, Btr, seed + 10);
    aV = nearest(CV, Xs_te);
    res.validity_only = ev(Tv(aV,:), PKv(aV,:));
end
res.validity_only.groups = best.K;
res.proposed_state = ev(Tk(aS,:), PKk(aS,:)); res.proposed_state.groups = K;
Mtr = measurable_features(M, Str); Mte = measurable_features(M, Ste);
if K > 1, aK = knn_predict(stdz(Mtr, Mtr), lab, stdz(Mtr, Mte), 5); else, aK = ones(nTe, 1); end
res.proposed_knn = ev(Tk(aK,:), PKk(aK,:)); res.proposed_knn.groups = K;
res.proposed_knn.acc = mean(aK == aS);
envFlag = any(LMte > 1.05*PKk(aK,:)/M.OLF, 2);                     % envelope
Te = Tk(aK,:); PKe = PKk(aK,:);
Te(envFlag,:) = repmat(T1, sum(envFlag), 1); PKe(envFlag,:) = repmat(pk1, sum(envFlag), 1);
res.proposed_knn_env = ev(Te, PKe); res.proposed_knn_env.groups = K + (K > 1);
res.proposed_knn_env.fallback_pct = 100*mean(envFlag);
% (d) scalar aggregate-DER index: prior-study resolution and K bins
[Ts, PKs, ~, bte] = scalar_groups(M, 11, Str, Ste, LMtr, Ptr, Btr, seed + 40);
res.scalar_11 = ev(Ts(bte,:), PKs(bte,:)); res.scalar_11.groups = 11;
[Ts, PKs, ~, bte] = scalar_groups(M, K, Str, Ste, LMtr, Ptr, Btr, seed + 60);
res.scalar_K = ev(Ts(bte,:), PKs(bte,:)); res.scalar_K.groups = K;
res.K = K; res.method = meth; res.K_validity = best.K; res.ungradable_validity = ungV;
res.sil = best.sil; res.db = best.db;
res.sizes = accumarray(lab, 1)';
ins = evaluate(M, Tk(lab,:), PKk(lab,:), LFtr, Ptr, Btr); res.insample_total = ins.total;
% (e) optional group-count sweep: [kind(1=cluster,2=scalar), n, in-sample %, out-of-sample %]
if doSweep
    sw = [];
    for Kx = [2 3 4 5 6 8]
        lx = cluster_labels(Xs_tr, 'kmeans', Kx); Kr = max(lx);
        Cx = zeros(Kr, size(Xs_tr, 2)); for c = 1:Kr, Cx(c,:) = mean(Xs_tr(lx == c, :), 1); end
        [Tx, PKx] = group_settings(M, lx, Kr, LMtr, Ptr, Btr, seed + 300 + 10*Kx);
        ax = nearest(Cx, Xs_te);
        ei = evaluate(M, Tx(lx,:), PKx(lx,:), LFtr, Ptr, Btr); eo = ev(Tx(ax,:), PKx(ax,:));
        sw = [sw; 1, Kx, ei.affected, eo.affected]; %#ok<AGROW>
    end
    for nb = [2 3 4 6 8 11]
        [Ts, PKs, btr, bte] = scalar_groups(M, nb, Str, Ste, LMtr, Ptr, Btr, seed + 600 + 20*nb);
        ei = evaluate(M, Ts(btr,:), PKs(btr,:), LFtr, Ptr, Btr); eo = ev(Ts(bte,:), PKs(bte,:));
        sw = [sw; 2, nb, ei.affected, eo.affected]; %#ok<AGROW>
    end
    res.sweep = sw;
end
end

function S = summarise(reps, names, caseName, doSweep)
sdv = @(x) (numel(x) > 1)*std(x);
fprintf('\n%-18s %6s %18s %16s %16s %12s %10s\n', 'scheme', 'groups', 'test events', ...
    'affected %', 't_prim (s)', 'trunk pk (A)', 'min marg');
for k = 1:numel(names)
    nm = names{k};
    g = cellfun(@(x) x.(nm).groups, reps); tot = cellfun(@(x) x.(nm).total, reps);
    aff = cellfun(@(x) x.(nm).affected, reps); tp = cellfun(@(x) x.(nm).t_prim, reps);
    pk = cellfun(@(x) x.(nm).pk_trunk, reps); mm = cellfun(@(x) x.(nm).min_margin, reps);
    fprintf('%-18s %6.1f %9.1f +/- %5.1f %8.2f +/- %5.2f %7.4f +/- %.4f %8.1f %10.3f\n', nm, ...
        mean(g), mean(tot), sdv(tot), mean(aff), sdv(aff), mean(tp), sdv(tp), mean(pk), min(mm));
    S.(nm) = struct('groups', mean(g), 'total_mean', mean(tot), 'total_std', sdv(tot), ...
        'affected_mean', mean(aff), 'affected_std', sdv(aff), 't_prim_mean', mean(tp), ...
        't_prim_std', sdv(tp), 'pk_trunk_mean', mean(pk), 'min_margin', min(mm));
end
S.K_selected = cellfun(@(x) x.K, reps)';
S.K_validity = cellfun(@(x) x.K_validity, reps)';
S.ungradable_validity = cellfun(@(x) x.ungradable_validity, reps)';
S.insample_total = cellfun(@(x) x.insample_total, reps)';
S.knn_acc_mean = mean(cellfun(@(x) x.proposed_knn.acc, reps));
fprintf('K selected: %s   validity-best K: %s (ungradable groups %s)   in-sample events (proposed): %s   kNN accuracy: %.3f\n', ...
    mat2str(S.K_selected), mat2str(S.K_validity), mat2str(S.ungradable_validity), mat2str(S.insample_total), S.knn_acc_mean);
if doSweep
    SW = cat(3, cellfun(@(x) x.sweep, reps, 'UniformOutput', false));
    SW = mean(cat(3, SW{:}), 3); S.sweep = SW;
    fprintf('sweep [kind n in%% out%%] (mean over replications):\n'); disp(round(SW*10)/10);
end
here = fileparts(mfilename('fullpath'));
save(fullfile(here, sprintf('results_%s.mat', caseName)), 'S', 'reps', '-v7');
end
