#! /bin/bash

mkdir -p igb-dataset-full
cd igb-dataset-full
mkdir -p processed
cd processed

echo "IGBH600M (Heteregeneous) download starting"

# paper
mkdir -p paper
cd paper
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_feat.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_19.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/paper_id_index_mapping.npy
cd ..

# paper__cites__paper
mkdir -p paper__cites__paper
cd paper__cites__paper
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__cites__paper/edge_index.npy
cd ..

# author
mkdir -p author
cd author
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/author/author_id_index_mapping.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/author/node_feat.npy
cd ..

# conference
mkdir -p conference
cd conference
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/conference/conference_id_index_mapping.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/conference/node_feat.npy
cd ..

# institute
mkdir -p institute
cd institute
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/institute/institute_id_index_mapping.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/institute/node_feat.npy
cd ..

# journal
mkdir -p journal
cd journal
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/journal/journal_id_index_mapping.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/journal/node_feat.npy
cd ..

# fos
mkdir -p fos
cd fos
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/fos/fos_id_index_mapping.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/fos/node_feat.npy
cd ..

# author__affiliated_to__institute
mkdir -p author__affiliated_to__institute
cd author__affiliated_to__institute
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/author__affiliated_to__institute/edge_index.npy
cd ..

# paper__published__journal
mkdir -p paper__published__journal
cd paper__published__journal
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__published__journal/edge_index.npy
cd ..

# paper__topic__fos
mkdir -p paper__topic__fos
cd paper__topic__fos
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__topic__fos/edge_index.npy
cd ..

# paper__venue__conference
mkdir -p paper__venue__conference
cd paper__venue__conference
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__venue__conference/edge_index.npy
cd ..

# paper__written_by__author
mkdir -p paper__written_by__author
cd paper__written_by__author
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__written_by__author/edge_index.npy
cd ..

cd ../..

echo "IGBH-IGBH (Heteregeneous) download complete"


num_paper_nodes = 269346174
paper_node_features = np.memmap('/home/ubuntu/dgl/examples/graphbolt/rgcn/igb_dataset/igb_dataset_full/processed/paper/node_label_19.npy', dtype='float32', mode='r',  shape=(num_paper_nodes,1))
num_paper_nodes = 48521486
paper_node_features = np.memmap('/home/ubuntu/dgl/examples/graphbolt/rgcn/datasets/igb-dataset-full-seeds/edges/author__affiliated_to__institute.npy', dtype='int32', mode='r',  shape=(num_paper_nodes,2))
