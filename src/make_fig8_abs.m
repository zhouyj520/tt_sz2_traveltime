% three_dat_cube_faces_equalbox.m
% 读取 .json / .dat / .dat.sz(.out) 三个体，计算 RMSE/MAE/PSNR，
% 绘制“立方体三面切片”三联图（原始/解压/绝对误差）。
% 将本文件与 TTP_overthrust_20_single.json/.dat/.dat.sz(.out) 放同一目录直接运行。

clear; clc;

%% ===================== 用户参数 =====================
json_file = 'TTP_overthrust_20_single.json';
orig_dat  = 'TTP_overthrust_20_single.dat';
dec_dat   = 'TTP_overthrust_20_single.dat.sz.out';   % 若不存在会自动尝试 .sz
out_png   = 'traveltime_comparison_solid.png';
dpi       = 300;

DS_FACTOR      = 1;          % 降采样因子（1=不降采样；2或4更快）
ERR_CAX_PCT    = 99.0;       % 误差色轴上限分位数（抑制尖峰）
FACE_POS       = {'x',1; 'y',1; 'z','end'};  % 三张面：x=1、y=1、z=end（顶/前/左）
NICE_EQUAL_BOX = true;       % true=视觉等长（立方体外观）；false=物理等比
CB_SHRINK      = 0.70;       % 颜色条高度相对于子图高度的比例（0~1，越小越短）

%% ===================== 读取元数据与数据 =====================
fprintf('加载元数据: %s\n', json_file);
[shape, dtype, extents] = load_meta(json_file);   % shape=[nx ny nz]
nx = shape(1); ny = shape(2); nz = shape(3);
fprintf('体素数: %d x %d x %d = %.3g\n', nx, ny, nz, double(nx)*ny*nz);
bytes_per = 4; if strcmpi(dtype,'double'), bytes_per = 8; end
fprintf('理论内存（单体）≈ %.2f GB\n', double(nx)*ny*nz*bytes_per/1024^3);

fprintf('加载原始体: %s\n', orig_dat);
vol0 = load_dat_bin(orig_dat, [nx ny nz], dtype);

if ~isfile(dec_dat)
    alt = regexprep(dec_dat, '\.sz\.out$', '.sz');
    if isfile(alt), fprintf('[INFO] 用 %s 代替\n', alt); dec_dat = alt; end
end
fprintf('加载解压体: %s\n', dec_dat);
vol1 = load_dat_bin(dec_dat, [nx ny nz], dtype);

% 单精度省内存
vol0 = single(vol0); vol1 = single(vol1);

%% ===================== 指标 =====================
mask  = ~(isnan(vol0) | isnan(vol1) | isinf(vol0) | isinf(vol1));
diffv = vol1 - vol0;
abser = abs(diffv);
N     = nnz(mask);
rmse  = sqrt(sum((diffv(mask)).^2)/max(N,1));
mae   = sum(abser(mask))/max(N,1);
vmax0 = max(abs(vol0(mask)));
psnr  = 20*log10( vmax0 / max(rmse, eps) );
fprintf('RMSE = %.6g, MAE = %.6g, PSNR = %.2f dB\n', rmse, mae, psnr);

%% ===================== 降采样（可选） =====================
if DS_FACTOR > 1
    fprintf('[DS] 降采样因子 = %d ...\n', DS_FACTOR);
    vol0  = vol0(1:DS_FACTOR:end, 1:DS_FACTOR:end, 1:DS_FACTOR:end);
    vol1  = vol1(1:DS_FACTOR:end, 1:DS_FACTOR:end, 1:DS_FACTOR:end);
    abser = abser(1:DS_FACTOR:end, 1:DS_FACTOR:end, 1:DS_FACTOR:end);
    [nx,ny,nz] = size(vol0); shape = [nx ny nz];
    fprintf('[DS] 新尺寸: %d x %d x %d\n', nx, ny, nz);
end

%% ===================== 色轴与坐标 =====================
vmin = min(vol0(:)); vmax = max(vol0(:));
emax = max(abser(:));
ecax = prctile(abser(:), ERR_CAX_PCT); ecax = max(ecax, eps);

% 物理坐标（JSON 无 extent_m 时，默认各向同性：extents=dims）
Lx = extents(1); Ly = extents(2); Lz = extents(3);
x = linspace(0, Lx, size(vol0,1));
y = linspace(0, Ly, size(vol0,2));
z = linspace(0, Lz, size(vol0,3));

% 显示为 km
x_km = x / 1000;  y_km = y / 1000;  z_km = z / 1000;

% 为 slice 的 [Y X Z] 顺序，交换 X/Y 维
V0 = permute(vol0,[2 1 3]);   % [Y X Z]
V1 = permute(vol1,[2 1 3]);
VE = permute(abser,[2 1 3]);

%% ===================== 三联图（立方体三面切片） =====================
fig = figure('Color','w','Position',[100 90 1680 560]);

ax1 = subplot(1,3,1);
cb1 = plot_cubefaces(ax1, V0, x_km, y_km, z_km, FACE_POS, [vmin vmax], 'Original', 'parula');

ax2 = subplot(1,3,2);
cb2 = plot_cubefaces(ax2, V1, x_km, y_km, z_km, FACE_POS, [vmin vmax], 'Decompressed (ABS=1e-5)', 'parula');

ax3 = subplot(1,3,3);
cb3 = plot_cubefaces(ax3, VE, x_km, y_km, z_km, FACE_POS, [0 ecax], 'Absolute Error (ABS=1e-5)', 'hot');

% 统一外观：视觉等长 or 物理等比
for ax = [ax1,ax2,ax3]
    grid(ax,'on'); box(ax,'on'); axis(ax,'tight'); axis(ax,'vis3d');
    set(ax,'ActivePositionProperty','position');   % 固定轴位，避免 colorbar 挤压
    if NICE_EQUAL_BOX
        pbaspect(ax,[1 1 1]); daspect(ax,'auto'); set(ax,'Projection','perspective');
    else
        daspect(ax,[1 1 1]); pbaspect(ax,'auto'); set(ax,'Projection','orthographic');
    end
    set(ax,'LineWidth',0.8,'FontSize',10);
end
%view(ax1,[45 25]);  % 统一视角

% === 布局完成后，最后统一缩短并垂直居中每个颜色条 ===
shrink_cb(cb1, ax1, CB_SHRINK);
shrink_cb(cb2, ax2, CB_SHRINK);
shrink_cb(cb3, ax3, CB_SHRINK);

%% ===================== 导出 =====================
fprintf('导出：%s\n', out_png);
print(fig, out_png, '-dpng', sprintf('-r%d', dpi));
savefig(fig, strrep(out_png,'.png','.fig'));
fprintf('完成。\n');

%% ===================== 本地函数 =====================
function [dims, dtype, extents] = load_meta(json_path)
    meta = jsondecode(fileread(json_path));
    if isfield(meta,'dims_for_sz'), dims = meta.dims_for_sz(:).';
    elseif isfield(meta,'dims'),     dims = meta.dims(:).';
    else, error('JSON 中未找到 dims/dims_for_sz 字段。');
    end
    if isfield(meta,'dtype')
        switch lower(string(meta.dtype))
            case {'float','float32','single'}, dtype = 'single';
            case {'double','float64'},         dtype = 'double';
            otherwise, error('不支持的 dtype: %s', meta.dtype);
        end
    else
        dtype = 'single';
    end
    if isfield(meta,'extent_m')
        extents = double(meta.extent_m(:)).';
    else
        extents = double(dims);   % 无物理范围时，默认各向同性
    end
end

function vol = load_dat_bin(path, shape, dtype)
    nx = shape(1); ny = shape(2); nz = shape(3);
    fid = fopen(path,'r','ieee-le'); if fid<0, error('无法打开文件: %s', path); end
    c = onCleanup(@() fclose(fid));
    vol = fread(fid, nx*ny*nz, ['*' dtype]);
    if numel(vol) ~= nx*ny*nz
        error('文件长度不匹配：读到 %d，期望 %d', numel(vol), nx*ny*nz);
    end
    vol = reshape(vol, [nx ny nz]);
end

function cb = plot_cubefaces(ax, V, x, y, z, face_pos, climits, title_str, cmap_name)
    % V 维度为 [Y X Z]（主程序里已 permute）
    axes(ax); hold(ax,'on');

    % 索引网格（slice 用索引坐标；物理刻度由 ticklabel 提供）
    [Yg, Xg, Zg] = ndgrid(1:numel(y), 1:numel(x), 1:numel(z));

    % 解析三面位置
    xs = []; ys = []; zs = [];
    for k = 1:size(face_pos,1)
        axn = face_pos{k,1}; pos = face_pos{k,2};
        switch axn
            case 'x', xs = [xs, pos_if_end(pos, numel(x))]; % 第二维=X
            case 'y', ys = [ys, pos_if_end(pos, numel(y))]; % 第一维=Y
            case 'z', zs = [zs, pos_if_end(pos, numel(z))];
        end
    end

    % 三面切片
    hs = slice(ax, Xg, Yg, Zg, V, xs, ys, zs); %#ok<NASGU>
    set(hs, 'EdgeColor','none', 'FaceAlpha', 1.0);

    % km 刻度（只改标签）
    xticks(ax, linspace(1, numel(x), 5)); xticklabels(ax, round(linspace(x(1), x(end), 5), 2));
    yticks(ax, linspace(1, numel(y), 5)); yticklabels(ax, round(linspace(y(1), y(end), 5), 2));
    zticks(ax, linspace(1, numel(z), 5)); zticklabels(ax, round(linspace(z(1), z(end), 5), 2));

    xlabel(ax,'X (km)'); ylabel(ax,'Y (km)'); zlabel(ax,'Z (km)');

    % 颜色与色轴
    colormap(ax, cmap_name);
    caxis(ax, climits);
    cb = colorbar(ax);
    if contains(lower(title_str), 'error')
        ylabel(cb, '|Δt| (s)');
    else
        ylabel(cb, 't (s)');
    end

    title(ax, title_str, 'FontSize', 13);
    axis(ax,'tight'); view(ax,3);
end

function v = pos_if_end(p, n)
    if ischar(p) || isstring(p)
        if strcmpi(string(p),'end'), v = n; else, v = str2double(p); end
    else
        v = p;
    end
end

function shrink_cb(cb, ax, ratio)
    % 将颜色条高度设为子图高度的 ratio，并垂直居中
    if nargin<3, ratio = 0.7; end
    set([ax cb], 'Units','normalized');  % 统一坐标系
    axpos = get(ax,'Position');          % [left bottom width height]
    cbpos = get(cb,'Position');          % [left bottom width height]
    newH  = axpos(4) * ratio;
    cbpos(2) = axpos(2) + (axpos(4) - newH)/2;  % 居中
    cbpos(4) = newH;
    set(cb,'Position',cbpos);
end
