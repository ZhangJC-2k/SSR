clear all
close all
Num_scene = 10;%test scene num
Filepath1 = 'C:\Code_DPU\Result\';%DPU file path
Filepath2 = 'C:\Code_DPU\Label\';

psnr = 0;
PSNR = zeros(1,Num_scene);
Ssim = zeros(1,Num_scene);
for n=1:Num_scene
    load(strcat(Filepath1,num2str(n),'.mat'));
    load(strcat(Filepath2,num2str(n),'.mat'));
    mse = mean(mean((hsi-label).^2));
    PSNR(1,n) = mean(10*log10(1./mse));
    psnr = psnr + PSNR(1,n);
    ssim=cal_ssim(double(im2uint8(hsi)), double(im2uint8(label)),0,0);
    Ssim(n) = ssim;
end
psnr = psnr/Num_scene;
PSNR =PSNR';
Ssim =Ssim';
ssim = mean(Ssim);



