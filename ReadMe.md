python train_wo_feat.py --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 --dropout 0.6 --num_neg 6 --epochs 500 --prop_step 2 --metric hits@20 --residual 0.1

python train_w_feat.py --dataset ogbl-collab --lr 0.0004 --hidden 512 --batch_size 16384 --dropout 0.2 --num_neg 6 --epochs 500 --prop_step 6 --metric hits@50

python train_wo_feat.py --dataset ogbl-ppa --lr 0.001 --hidden 512 --batch_size 65536 --dropout 0.2 --num_neg 6 --epochs 500 --prop_step 2 --metric hits@100 --residual 0.1

python train_w_feat.py --dataset ogbl-citation2 --lr 0.003 --hidden 128 --batch_size 131072 --dropout 0 --num_neg 6 --epochs 200 --prop_step 3 --metric MRR

to run this model, you only need to have pytorch, dgl, ogb, matplotlib installed.
we also have a pyg version for the model.you only need to remove the --metric option to run the code.