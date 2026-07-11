# train DenseFuse
python train.py --device 2 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-DenseFuse.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_DenseFuse --exist-ok
# train PSFusion
python train.py --device 2 --weights yolov5s.pt --data data/sota_fusions/m3fd-sotafusion-PSFusion.yaml --hyp data/hyps/hyp.scratch-low.yaml --epochs 36 --batch-size 32 --img 640 --name m3fd_zxSplit_sotafusion_v5small_e36_bs32_PSFusion --exist-ok
