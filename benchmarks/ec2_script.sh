export INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
cd ~/git_repo/benchmarks
python replace_branch.py --branch ${GIT_BRANCH}
bash publish.sh ${INSTANCE_TYPE} cpu
aws s3 sync results s3://dgl-asv-data/ci/results/
aws s3 sync html s3://dgl-asv-data/ci/html/ --acl public-read
