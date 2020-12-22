export INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
cd ~/git_repo/benchmarks
python replace_branch.py --branch ${GIT_BRANCH}
DEVICE=cpu
if [[ ${INSTANCE_TYPE} == g* ]] || [[ ${INSTANCE_TYPE} == p* ]]; then
    DEVICE=gpu
else
    DEVICE=cpu
fi
bash publish.sh ${INSTANCE_TYPE} ${DEVICE}
aws s3 sync results s3://dgl-asv-data/ci/results/
aws s3 sync html s3://dgl-asv-data/ci/html/ --acl public-read
