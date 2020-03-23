import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', type=str, default='')
    parser.add_argument('--vocab', type=str, default='')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--portion', type=float, default=0.01)
    return parser.parse_args()

args = parse_args()

times = 10
emb = args.emb
vocab = args.vocab
label = args.label
portion = args.portion
output = "result.txt"
workspace = "workspace/"

os.system("rm -rf result.txt")
os.system("touch result.txt")

os.system("./program/preprocess -vocab %s -vector %s -label %s -output %s -debug 2 -binary 1 -times %d -portion %f"\
          % (vocab, emb, label, workspace, times, portion))

for i in xrange(times):
    os.system("./liblinear/train -s 0 -q %strain%d %smodel%d" % (workspace, i, workspace, i))

for i in xrange(times):
    os.system("./liblinear/predict -b 1 -q %stest%d %smodel%d %spredict%d" % (workspace, i, workspace, i, workspace, i))

for i in xrange(times):
    os.system("./program/score -predict %spredict%d -candidate %scan%d >> %s" % (workspace, i, workspace, i, output))


#os.system("./run.sh %s %s %s" % (args.emb, args.vocab, args.label))
os.system("python3 score.py result.txt")
