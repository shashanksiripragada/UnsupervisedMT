#!/bin/bash
#SBATCH --job-name unsupTrain
#SBATCH --account shashanks
#SBATCH --partition long 
#SBATCH --nodes 1
#SBATCH --mem 45G
#SBATCH --gres gpu:4
#SBATCH --time UNLIMITED
#SBATCH --mail-type END
##SBATCH -w gnode

module load use.own
module add python/3.7.0
module add pytorch/1.0.0

set -x

REMOTE_DIR="ada:/share1/shashanks/UnsupervisedMT/NMT"
SSD_DIR=/ssd_scratch/cvit/shashanks/UnsupervisedMT/NMT
MONO_DIR=$SSD_DIR/data/mono
PARA_DIR=$SSD_DIR/data/para/dev
SRC_DIR=$SSD_DIR/src
TOOLS_DIR=$SSD_DIR/tools

mkdir -p $SSD_DIR $MONO_DIR $PARA_DIR $SRC_DIR $TOOLS_DIR

rsync -rvz $REM_DIR/src/ $SSD_DIR/src/
rsync -rvz $REM_DIR/tools/ $SSD_DIR/tools/
rsync -rvz $REM_DIR/main.py $SSD_DIR/
rsync -rvz $REM_DIR/preprocess.py $SSD_DIR/
rsync -rvz $REM_DIR/data/mono/all.{en,fr}.tok.60000.pth $MONO_DIR/
rsync -rvz $REM_DIR/data/para/ $PARA_DIR/
rsync -rvz $REM_DIR/data/mono/all.en-fr.60000.vec $MONO_DIR/

MONO_DATASET='en:./data/mono/all.en.tok.60000.pth,,;fr:./data/mono/all.fr.tok.60000.pth,,'
PARA_DATASET='en-fr:,./data/para/dev/newstest2013-ref.XX.60000.pth,./data/para/dev/newstest2014-fren-src.XX.60000.pth'
PRETRAINED='./data/mono/all.en-fr.60000.vec'

cd $SSD_DIR
pwd

python3 main.py \
	--exp_name test \
	--transformer True \
	--n_enc_layers 4 \
	--n_dec_layers 4 \
	--share_enc 3 \
	--share_dec 3 \
	--share_lang_emb True \
	--share_output_emb True \
	--langs 'en,fr' \
	--n_mono -1 \
	--mono_dataset $MONO_DATASET \
	--para_dataset $PARA_DATASET \
	--mono_directions 'en,fr' \
	--word_shuffle 3                          \
	--word_dropout 0.1                        \
	--word_blank 0.2                          \
	--pivo_directions 'en-fr-en,fr-en-fr'\
	--pretrained_emb $PRETRAINED               \
	--pretrained_out True                      \
	--lambda_xe_mono '0:1,100000:0.1,300000:0' \
	--lambda_xe_otfd 1                         \
	--otf_num_processes 30                     \
	--otf_sync_params_every 1000               \
	--enc_optimizer adam,lr=0.0001             \
	--group_by_size True                       \
	--batch_size 64                            \
	--epoch_size 100000                        \
	--stopping_criterion bleu_en_fr_valid,10   \
	--freeze_enc_emb False                     \
	--freeze_dec_emb False 

