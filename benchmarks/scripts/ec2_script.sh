export INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
export MOUNT_PATH=/mnt/efs/fs1/
cd ~/git_repo/benchmarks/scripts
python replace_branch.py --branch ${GIT_BRANCH}
DEVICE=${DEVICE:-cpu}
echo "DEVICE=$DEVICE"
echo "INSTANCE_TYPE=$INSTANCE_TYPE"
aws s3 sync s3://dgl-asv-data/ci/results/ ~/git_repo/benchmarks/results
bash publish.sh ${INSTANCE_TYPE} ${DEVICE}
aws s3 sync ~/git_repo/benchmarks/results s3://dgl-asv-data/ci/results/
aws s3 sync ~/git_repo/benchmarks/html s3://dgl-asv-data/ci/html/ --acl public-read
