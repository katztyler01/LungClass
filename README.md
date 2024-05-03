# LungClass

## Model Downloads
DreamBooth model downloaded and used from this link :<br> https://github.com/huggingface/diffusers/tree/main/examples/dreambooth <br>


CXR-CLIP model downloaded and used from this link :<br> https://github.com/kakaobrain/cxr-clip <br>
1.Environment setup: 
pip install -r requirements.txt<br>
2.Pre-Train model:<br> 
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=45678 train.py {--config-name default_config}<br>
3.Zero-shot Evaluation:<br> python evaluate_clip.py test.checkpoint=${CKPT_PATH/model-best.tar}<br>
4.Fine-tuned Classifier (linear probing):<br>
train<br>
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=rsna_pneumonia data_valid=rsna_pneumonia model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%<br>
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=rsna_pneumonia data_valid=rsna_pneumonia model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%<br>
5.evaluate<br>
python evaluate_finetune.py data_test=rsna_pneumonia test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}<br>



## Exploratory Data Analysis
The exploratory data analysis can be viewed an run in the notebook `init.ipynb`.

## Principal Component Analysis
The PCA implementation can be viewed and run in the notebook `PCA.ipynb`.

## Lasso Regression 
The Lasso regression implementation can be viewed and run in the notebook `LassoRegression.ipynb`.

## Logistic Regression
The logistic regression implementation can be viewed and run in the notebook `log_reg.ipynb`.

