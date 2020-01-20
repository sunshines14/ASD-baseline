#!/bin/bash

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License

stage=0

# pre-training (train)
if [ $stage -le 1 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py \
        --track=LA \
        --input_size=64000 \
        --feature=spect \
        --data_tag=0 \
        --train_batch_size=32 \
        --dev_batch_size=32 \
        --num_epochs=200 \
        --embedding_size=128 \
        --model_comment=siamesefit1 \
        --loss=nll \
        --optim=adam \
        --lr=0.00005 \
        --wd=0 \
        --sched_factor=0.1 \
        --sched_patience=50 \
        --sched_min_lr=0 
exit 0
fi

# pre-training (eval)
if [ $stage -le 2 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --track=LA \
        --input_size=64000 \
        --feature=spect \
        --data_tag=0 \
        --train_batch_size=32 \
        --dev_batch_size=32 \
        --num_epochs=200 \
        --embedding_size=128 \
        --model_comment=siamesefit1 \
        --loss=nll \
        --optim=adam \
        --lr=0.00005 \
        --wd=0 \
        --sched_factor=0.1 \
        --sched_patience=50 \
        --sched_min_lr=0 \
        --eval_mode \
        --eval_batch_size=32 \
        --eval_num_checkpoint=0
exit 0
fi

# pre-training (result)
if [ $stage -le 3 ]; then
    python3 eval.py \
        models/pre/model_LA_64000_spect_0_32_200_128_siamesefit1/0.result \
        ../CORPUS/data_logical/ASVspoof2019_LA_asv/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
        #../CORPUS/data_logical/ASVspoof2019_LA_asv/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
exit 0
fi

# fine-tuning (train)
if [ $stage -le 4 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py \
        --track=LA \
        --input_size=64000 \
        --feature=spect \
        --data_tag=0 \
        --train_batch_size=32 \
        --dev_batch_size=32 \
        --num_epochs=200 \
        --embedding_size=128 \
        --model_comment=siamesefit1 \
        --loss=cs \
        --optim=adam \
        --lr=0.00005 \
        --wd=0 \
        --sched_factor=0.1 \
        --sched_patience=50 \
        --sched_min_lr=0 \
        --fit_mode \
        --fit_num_checkpoint=0
exit 0
fi

# pre-training (eval)
if [ $stage -le 5 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --track=LA \
        --input_size=64000 \
        --feature=spect \
        --data_tag=0 \
        --train_batch_size=32 \
        --dev_batch_size=32 \
        --num_epochs=200 \
        --embedding_size=128 \
        --model_comment=siamesefit1 \
        --loss=nll \
        --optim=adam \
        --lr=0.00005 \
        --wd=0 \
        --sched_factor=0.1 \
        --sched_patience=50 \
        --sched_min_lr=0 \
        --fit_mode \
        --eval_mode \
        --eval_batch_size=32 \
        --eval_num_checkpoint=0
exit 0
fi

# fine-tuning (result)
if [ $stage -le 6 ]; then
    python3 eval.py \
        models/tune/model_LA_64000_spect_0_32_200_128_siamesefit1/0.result \
        ../CORPUS/data_logical/ASVspoof2019_LA_asv/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt
        #../CORPUS/data_logical/ASVspoof2019_LA_asv/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
exit 0
fi