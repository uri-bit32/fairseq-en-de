import argparse
import fileinput

from tqdm import tqdm
from fairseq import bleu
from fairseq.data import dictionary


import pdb

def main():
    parser = argparse.ArgumentParser(description=(
        'Extract back-translations from the stdout of fairseq-generate. '
        'If there are multiply hypotheses for a source, we only keep the first one. '
    ))
    parser.add_argument('--output', required=True, help='output prefix')
    parser.add_argument('--srclang', required=True, help='source language (extracted from H-* lines)')
    parser.add_argument('--tgtlang', required=True, help='target language (extracted from S-* lines)')
    parser.add_argument('--minlen', type=int, help='min length filter')
    parser.add_argument('--maxlen', type=int, help='max length filter')
    parser.add_argument('--ratio', type=float, help='ratio filter')
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()

    dict = dictionary.Dictionary()
    scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())


    
    def validate(src, tgt):
        srclen = len(src.split(' ')) if src != '' else 0
        tgtlen = len(tgt.split(' ')) if tgt != '' else 0
        if (
            (args.minlen is not None and (srclen < args.minlen or tgtlen < args.minlen))
            or (args.maxlen is not None and (srclen > args.maxlen or tgtlen > args.maxlen))
            or (args.ratio is not None and (max(srclen, tgtlen) / float(min(srclen, tgtlen)) > args.ratio))
        ):
            return False
        return True

    def safe_index(toks, index, default):
        try:
            return toks[index]
        except IndexError:
            return default
#    pdb.set_trace()
    with open(args.output + '.' + args.srclang, 'w') as src_h, \
            open(args.output + '.' + args.tgtlang, 'w') as tgt_h:
        for line in tqdm(fileinput.input(args.files)):
            if line.startswith('S-'):
#                pdb.set_trace()
                tgt = safe_index(line.rstrip().split('\t'), 1, '')
            elif line.startswith('T-'):
                gt = safe_index(line.rstrip().split('\t'), 1, '')
            elif line.startswith('H-'):
                if tgt is not None:
                   if gt is not None:
#                    pdb.set_trace()
                    src = safe_index(line.rstrip().split('\t'), 2, '')
#                    sent_score = -float(safe_index(line.rstrip().split('\t'), 1, ''))
#                    print(str(sent_score) + "#####" + src)
#                    scorer.reset(one_init=True)
#                    gt_tok = dict.encode_line(gt)
#                    tgt_tok = dict.encode_line(tgt)
#                    scorer.add(gt_tok, tgt_tok)
#                    out = scorer.result_string(1)                      
#                    print(out+' ' + src + ' ' + gt)                  
#                    pdb.set_trace()
#                    out = float(out.split(' ')[2].split(',')[0])
#                    out = float(out.split('/')[2])
#                    if out < 5:
#                       tgt = gt
                    if validate(src, tgt):
                        print(src, file=src_h)
                        print(tgt, file=tgt_h)
                    else:
                        if validate(gt, tgt):
                           print(gt, file=src_h)
                           print(tgt, file=tgt_h)
#                        print("##" + str(max(len(tgt.split(' ')), len(gt.split(' '))) / float(min(len(tgt.split(' ')), len(gt.split(' ')))))    + "##" + str(max(len(tgt.split(' ')), len(src.split(' '))) / float(min(len(tgt.split(' ')), len(src.split(' '))))))
#                        print(src, file=src_h)
#                        print(gt, file=tgt_h)
                    tgt = None
                    gt = None


if __name__ == '__main__':
    main()
