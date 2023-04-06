eval "$(conda shell.bash hook)"

set -e

DECA_DIR='/home/anuragr/Research/DECA'
FACELIT_DIR='/home/anuragr/Research/facelit'
DEEP3D_DIR='/home/anuragr/Research/Deep3DFaceRecon_pytorch'

prepare_deca () {
    conda activate facelit; python gen_samples.py --dataset $2 --network $FACELIT_DIR/pretrained/$1.pkl --seeds 0-1048 --trunc 1.0 --outdir $FACELIT_DIR/pretrained/eval/$1_deca --deca-samples True 
    cd $DECA_DIR
    conda activate deca; python demos/demo_reconstruct.py -i $FACELIT_DIR/pretrained/eval/$1_deca/img/ --saveCode True --saveVis False -s $FACELIT_DIR/pretrained/eval/$1_deca/deca_fits --iscrop False --rasterizer_type pytorch3d
    cd $FACELIT_DIR/facelit
}


report_metrics_id () {
    conda activate facelit; python gen_samples.py --dataset $2 --network $FACELIT_DIR/pretrained/$1.pkl --seeds 0-1048 --trunc 1.0 --outdir $FACELIT_DIR/pretrained/eval/$1_id --double-views True
    CUDA_VISIBLE_DEVICES=0 conda activate facelit; python calc_face_consistency.py --img-dir $FACELIT_DIR/pretrained/eval/$1_id
}

prepare_face_recon () {
    conda activate facelit; python prepare_face_landmarks.py --data_dir $FACELIT_DIR/pretrained/eval/$1_deca/img
    cd $DEEP3D_DIR
    conda activate deep3d_pytorch; python estimate_pose_gmpi.py --name face_recon --epoch 20 --gmpi_img_root $FACELIT_DIR/pretrained/eval/$1_deca/img/ --gmpi_detect_root $FACELIT_DIR/pretrained/eval/$1_deca/img/detections/ --gmpi_depth_root $FACELIT_DIR/pretrained/eval/$1_deca/label/
    cd $FACELIT_DIR/facelit
}

report_metrics_3d () {
    prepare_deca $1 $2
    prepare_face_recon $1
    conda activate facelit; python calc_deca_consistency.py --eval-dir $FACELIT_DIR/pretrained/eval/$1_deca/
    conda activate facelit; python calc_geometry_metrics.py --geo_dir $FACELIT_DIR/pretrained/eval/$1_deca 
}

report_metrics_fid_kid () {
    conda activate facelit; python calc_metrics.py --metrics=fid50k_full,kid50k_full --network=$FACELIT_DIR/pretrained/$1.pkl --data=../data/$2 --light_cond=True --verbose=False 
}


report_all_metrics () {
    echo Reporting Metrics for $1

    report_metrics_3d $1_full $2
    report_metrics_id $1_full $2
    report_metrics_fid_kid $1_full $3

    report_metrics_3d $1_diffuse $2
    report_metrics_id $1_diffuse $2
    report_metrics_fid_kid $1_diffuse $3
    echo ===================================

}

# report_all_metrics network_name dataset_name dataset_directory

report_all_metrics ffhq_128 ffhq FFHQ2x_512.zip
report_all_metrics ffhq ffhq FFHQ2x_512.zip
report_all_metrics metfaces metfaces MetFaces_512.zip
report_all_metrics celeba_hq celeba_hq CelebA_HQ_512.zip



