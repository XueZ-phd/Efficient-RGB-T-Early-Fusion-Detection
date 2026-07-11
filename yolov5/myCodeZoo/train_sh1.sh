# train DDcGAN
python train.py --device 1 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-DDcGAN.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_DDcGAN --exist-ok
# train DIVFusion
python train.py --device 1 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-DIVFusion.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_DIVFusion --exist-ok
