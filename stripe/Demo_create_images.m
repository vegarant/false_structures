
dest = './images';
if (exist(dest) ~= 7)
    mkdir(dest);
end

n             = 32;
line_width    = 3;
a             = 0.04;


f = @(image) (image+a)/(1+2*a);

vmax = 1+a;
vmin = 1-a;

% Do horizontal images first
lh1 = 3;
lh2 = 18;

im1h = -a*ones([n,n]);
im1h_false = a*ones([n,n]);

im2h = -a*ones([n,n]);
im2h_false = a*ones([n,n]);

idx1h = lh1:lh1+line_width-1;
idx2h = lh2:lh2+line_width-1;

im1h(idx1h,:) = im1h(idx1h, :) + 1;
im1h_false(idx1h,:) = im1h_false(idx1h, :) + 1;

im2h(idx2h,:) = im1h(idx2h, :) + 1;
im2h_false(idx2h,:) = im1h_false(idx2h, :) + 1;
fname1h = sprintf('im1h_n_%d.png', n);
fname1h_false = sprintf('im1h_false_n_%d.png', n);
fname2h = sprintf('im2h_n_%d.png', n);
fname2h_false = sprintf('im2h_false_n_%d.png', n);

im1h_u8       = im2uint8(f(im1h));
im1h_u8_false = im2uint8(f(im1h_false));
im2h_u8       = im2uint8(f(im2h));
im2h_u8_false = im2uint8(f(im2h_false));

imwrite(im1h_u8, fullfile(dest,fname1h));
imwrite(im1h_u8_false, fullfile(dest,fname1h_false));
imwrite(im2h_u8, fullfile(dest,fname2h));
imwrite(im2h_u8_false, fullfile(dest,fname2h_false));

lv1 = 3;
lv2 = 18;

im1v = a*ones([n,n]);
im1v_false = -a*ones([n,n]);

im2v = a*ones([n,n]);
im2v_false = -a*ones([n,n]);

idx1v = lv1:lv1+line_width-1;
idx2v = lv2:lv2+line_width-1;

im1v(:,idx1v) = im1v(:,idx1v) + 1;
im1v_false(:,idx1v) = im1v_false(:,idx1v) + 1;

im2v(:,idx2v) = im1v(:,idx2v) + 1;
im2v_false(:,idx2v) = im1v_false(:,idx2v) + 1;
fname1v = sprintf('im1v_n_%d.png', n);
fname1v_false = sprintf('im1v_false_n_%d.png', n);
fname2v = sprintf('im2v_n_%d.png', n);
fname2v_false = sprintf('im2v_false_n_%d.png', n);

im1v_u8       = im2uint8(f(im1v));
im1v_u8_false = im2uint8(f(im1v_false));
im2v_u8       = im2uint8(f(im2v));
im2v_u8_false = im2uint8(f(im2v_false));

imwrite(im1v_u8, fullfile(dest,fname1v));
imwrite(im1v_u8_false, fullfile(dest,fname1v_false));
imwrite(im2v_u8, fullfile(dest,fname2v));
imwrite(im2v_u8_false, fullfile(dest,fname2v_false));

