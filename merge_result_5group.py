import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result1', default=None, type=str)
    parser.add_argument('--result2', default=None, type=str)
    parser.add_argument('--result3', default=None, type=str)
    parser.add_argument('--result4', default=None, type=str)
    parser.add_argument('--result5', default=None, type=str)
    parser.add_argument('--output_json', default=None, type=str)

    args = parser.parse_args()

    assert args.result1 is not None
    assert args.result2 is not None
    assert args.result3 is not None
    assert args.result4 is not None
    assert args.result5 is not None

    with open(args.result1) as f:
        part1 = json.load(f)
    with open(args.result2) as f:
        part2 = json.load(f)
    with open(args.result3) as f:
        part3 = json.load(f)
    with open(args.result4) as f:
        part4 = json.load(f)
    with open(args.result5) as f:
        part5 = json.load(f)
    all_label = []
    with open('../dataset/label_id.txt', 'r') as f:
        for line in f.readlines():
            label, _ = line.split('\t')
            all_label.append(label)
    all_label_set = set(all_label)
    for file in part1.keys():
        label_set = set(part1[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part1[file]['result'][0]['labels'] += res_label
        part1[file]['result'][0]['scores'] += ['0'] * 32
        assert len(res_label) == 32
        assert len(part1[file]['result'][0]['labels']) == 82
    for file in part2.keys():
        label_set = set(part2[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part2[file]['result'][0]['labels'] += res_label
        part2[file]['result'][0]['scores'] += ['0'] * 32
        assert len(res_label) == 32
        assert len(part2[file]['result'][0]['labels']) == 82
    for file in part3.keys():
        label_set = set(part3[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part3[file]['result'][0]['labels'] += res_label
        part3[file]['result'][0]['scores'] += ['0'] * 32
        assert len(res_label) == 32
        assert len(part3[file]['result'][0]['labels']) == 82
    for file in part4.keys():
        label_set = set(part4[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part4[file]['result'][0]['labels'] += res_label
        part4[file]['result'][0]['scores'] += ['0'] * 32
        assert len(res_label) == 32
        assert len(part4[file]['result'][0]['labels']) == 82
    for file in part5.keys():
        label_set = set(part5[file]['result'][0]['labels'])
        res_label = list(all_label_set - label_set)
        part5[file]['result'][0]['labels'] += res_label
        part5[file]['result'][0]['scores'] += ['0'] * 32
        assert len(res_label) == 32
        assert len(part5[file]['result'][0]['labels']) == 82
    w1 = 0.2
    w2 = 0.2
    w3 = 0.2
    w4 = 0.2
    w5 = 0.2
    av_result = {}
    for file in part1.keys():
        av_result[file] = {}
        av_result[file]['result'] = []
        av_result[file]['result'].append({})
        av_result[file]['result'][0]['labels'] = []
        av_result[file]['result'][0]['scores'] = []
        for index, label in enumerate(part1[file]['result'][0]['labels']):
            av_result[file]['result'][0]['labels'].append(label)
            index1 = part1[file]['result'][0]['labels'].index(label)
            index2 = part2[file]['result'][0]['labels'].index(label)
            index3 = part3[file]['result'][0]['labels'].index(label)
            index4 = part4[file]['result'][0]['labels'].index(label)
            index5 = part5[file]['result'][0]['labels'].index(label)
            score = (w1 * eval(part1[file]['result'][0]['scores'][index1]) +
                     w2 * eval(part2[file]['result'][0]['scores'][index2]) +
                     w3 * eval(part3[file]['result'][0]['scores'][index3]) +
                     w4 * eval(part4[file]['result'][0]['scores'][index4]) +
                     w5 * eval(part5[file]['result'][0]['scores'][index5]))
            av_result[file]['result'][0]['scores'].append("%.4f" % score)
    for file in av_result.keys():
        score = av_result[file]['result'][0]['scores']
        label = av_result[file]['result'][0]['labels']
        sorted_score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
        idx = [i[0] for i in sorted_score]
        top20_index = idx[:50]
        av_result[file]['result'][0]['scores'] = [score[i] for i in top20_index]
        av_result[file]['result'][0]['labels'] = [label[i] for i in top20_index]
    with open(args.output_json, 'w', encoding="utf-8") as f:
        json.dump(av_result, f, ensure_ascii=False, indent=4)
