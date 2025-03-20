import os
import json
import random
import sys
import glob
import fasttext
import pandas as pd
from argparse import ArgumentParser

TOKENS_TO_REMOVE = ["<|user|>", "END", "Käänä suomeksi"]
FASTTEXT_LID_BINARY = "/scratch/project_462000444/zosaelai2/lid.176.bin"

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--translation_output_file', default="output.jsonl", type=str)
    ap.add_argument('--complete_preprocessed_file', default="output.jsonl", type=str)
    ap.add_argument('--final_output_file', default="output.jsonl", type=str)
    return ap

def detect_language(sent: str):
    lid_model = fasttext.load_model(FASTTEXT_LID_BINARY)
    # remove \n from sentences because fasttext processes by line
    sent = sent.replace("\n", " ") 
    pred = lid_model.predict(sent)
    # get top language
    lang = pred[0][0].split("__")[-1] 
    # get prob of top language
    prob = pred[1][0]
    return lang, prob

def check_compression(translations, ratio_threshold=0.3):
    print("--- check_compression ---")
    clean_translations = []
    for entry in translations:
        valid_entry = True
        for message in entry['messages']:
            text = message['content']
            if len(text) > 0:
                compressed = gzip.compress(text.encode('utf-8'))
                ratio = len(compressed) / len(text)
                if ratio < ratio_threshold:  # determined by observation
                    valid_entry = False
            else:
                valid_entry = False
        if valid_entry is True:
            clean_translations.append(entry)
    print("orig translations:", len(translations))
    print("clean translations:", len(clean_translations))
    return clean_translations

def check_untranslated_line(translated_text):
    print("--- check_untranslated_line ---")
    detected_lang, prob = detect_language(translated_text)
    if detected_lang == 'en':
        return False
    else:
        return True

def remove_extra_text(translated_text):
    print("---remove_extra_text---")
    for token_to_remove in TOKENS_TO_REMOVE:
        if token_to_remove in translated_text:   
            translated_text = translated_text[:translated_text.index(token_to_remove)].strip() 
    return translated_text



def main(argv):
    args = argparser().parse_args(argv[1:])
    df_all = pd.read_json(open(args.complete_preprocessed_file), lines=True)
    df_translate = pd.read_json(open(args.translation_output_file), lines=True)
    df_merged = pd.merge(df_all, df_translate, on=['sample_id', 'line_id'], how='left')
    df_merged['translation'] = df_merged.apply(lambda row: row['translation_y'] if pd.notnull(row['translation_y']) else row['translation_x'], axis=1)
    sample_ids = sorted(df_merged.sample_id.unique())
    with open(args.final_output_file, 'w') as outfile:
        for i, sample_id in enumerate(sample_ids):
            content = "\n".join(list(df_merged[df_merged.sample_id==sample_id].translation))
            entry = {'messages':[{'role':'user', 'content':content.strip()}]}
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Done! Processed {str(len(sample_ids))} samples.  Final output file written to {args.final_output_file}")
        outfile.close()




            


    


if __name__ == '__main__':
    sys.exit(main(sys.argv))
