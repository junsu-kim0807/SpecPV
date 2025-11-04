import os, json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="llama3.1-8b-instruct", choices=['llama3.1-8b-instruct','qwen3-8b'])
    args = parser.parse_args()

    files = os.listdir(f'outputs/{args.model_name}/longbench_v2/')
    output = ["Model\t\tOverall\t\tEasy\t\tHard\t\tShort\t\tMedium\t\tSingle\t\tMulti"]

    for file in files:

        filename = os.path.join('outputs/{args.model_name}/longbench_v2/', file)
        if not(filename.endswith(".jsonl")):
            continue
        try:
            pred_data = json.load(open(filename, encoding='utf-8'))
        except Exception as e:
            pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
        easy, hard, short, medium, single, multi = 0, 0, 0, 0, 0, 0
        easy_acc, hard_acc, short_acc, medium_acc, single_acc, multi_acc = 0, 0, 0, 0, 0, 0
        for pred in pred_data:
            acc = int(pred['judge'])
            if pred["difficulty"] == "easy":
                easy += 1
                easy_acc += acc
            else:
                hard += 1
                hard_acc += acc

            if pred['length'] == "short":
                short += 1
                short_acc += acc
            elif pred['length'] == "medium":
                medium += 1
                medium_acc += acc

            if (pred['domain'] == "Single-Document QA"):
                single += 1
                single_acc += acc
            elif (pred['domain'] == "Multi-Document QA"):
                multi += 1
                multi_acc += acc
        name = '.'.join(file.split('.')[:-1])
        output.append(name+'\t\t'+str(round(100*(easy_acc+hard_acc)/len(pred_data), 1))
                    +'\t\t'+str(round(100*easy_acc/easy, 1))+'\t\t'+str(round(100*hard_acc/hard, 1))
                    +'\t\t'+str(round(100*short_acc/short, 1))+'\t\t'+str(round(100*medium_acc/medium, 1))
                    +'\t\t'+str(round(100*single_acc/single, 1))+'\t\t'+str(round(100*multi_acc/multi, 1)))

    open('/home/lthpc/nvmessd/hcy/SpecPV/tests/results/qwen3/longbench_v2/result.txt', 'w', encoding='utf-8').write('\n'.join(output))
