DECA_DIR='/home/anuragr/Research/DECA'
DATA_DIR='/home/anuragr/Research/facelit/data/FFHQ'
cd $DECA_DIR
for d in $DATA_DIR; do
    echo "$d"
    save_path="${d/"FFHQ"/"FFHQ_deca_fits"}"
    echo "$save_path"
    python demos/demo_reconstruct.py -i $d --saveCode True --saveVis False -s $save_path --iscrop False
done