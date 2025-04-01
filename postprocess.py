import os
import json
import random
import sys
import glob
import gzip
import fasttext
import pandas as pd
from argparse import ArgumentParser

FASTTEXT_LID_BINARY = "/scratch/project_462000444/zosaelai2/lid.176.bin"
fasttext.FastText.eprint = lambda x: None
lid_model = fasttext.load_model(FASTTEXT_LID_BINARY)

TOKENS_TO_REMOVE = ["<|user|>", "END", "Käännä suomeksi" , "Translate into"]

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--translation_output_file', default="output.jsonl", type=str)
    ap.add_argument('--complete_preprocessed_file', default="output.jsonl", type=str)
    ap.add_argument('--final_output_file', default="output.jsonl", type=str)
    return ap

def detect_language(sent: str):
    # remove \n from sentences because fasttext processes by line
    sent = sent.replace("\n", " ") 
    pred = lid_model.predict(sent)
    # get top language
    lang = pred[0][0].split("__")[-1] 
    # get prob of top language
    prob = pred[1][0]
    return lang, prob

def check_compression(text, ratio_threshold=0.3):
    valid_text = True
    if len(text) > 0:
        compressed = gzip.compress(text.encode('utf-8'))
        ratio = len(compressed) / len(text)
        if ratio < ratio_threshold:  # determined by observation
            valid_text = False
    else:
        valid_text = False
    return valid_text

def check_untranslated_line(translated_text):
    #print("--- check_untranslated_line ---")
    detected_lang, prob = detect_language(translated_text)
    if detected_lang == 'en':
        return False
    else:
        return True

def check_untranslated_row(row):    
    detected_lang, prob = detect_language(row['translation'])
    if detected_lang == 'en':
        return False
    else:
        return True

def check_compression_row(row):    
    compression_ok = check_compression(row['translation'])
    return compression_ok

def remove_extra_text(translated_text):
    # print("---remove_extra_text---")
    for token_to_remove in TOKENS_TO_REMOVE:
        if token_to_remove in translated_text:   
            translated_text = translated_text[:translated_text.index(token_to_remove)].strip() 
    return translated_text

def remove_extra_text_in_translation_row(row):
    # print("---remove_extra_text_in_translation_row---")
    row['translation'] = remove_extra_text(row['translation'])
    return row

def main(argv):
    args = argparser().parse_args(argv[1:])
    df_all = pd.read_json(open(args.complete_preprocessed_file), lines=True)
    df_translate = pd.read_json(open(args.translation_output_file), lines=True)
    # remove extra text artifacts the Poro/Viking/Europa sometimes produces
    df_translate = df_translate.apply(remove_extra_text_in_translation_row, axis=1)
    # merge translations and the rest of the preprocessed data
    print("Merging dataframes")
    df_merged = pd.merge(df_all, df_translate, on=['sample_id', 'line_id'], how='left')
    df_merged['translation'] = df_merged.apply(lambda row: row['translation_y'] if pd.notnull(row['translation_y']) else row['translation_x'], axis=1)
    sample_ids = sorted(df_merged.sample_id.unique())
    print("Combining translated lines")
    df_final = {'translation':[]}
    for i, sample_id in enumerate(sample_ids):
        translation = "\n".join(list(df_merged[df_merged.sample_id==sample_id].translation))
        df_final['translation'].append(translation)
    df_final = pd.DataFrame.from_dict(df_final)
    print("Checking lang_id and compression")
    df_final['lang_id_ok'] = df_final.apply(check_untranslated_row, axis=1)
    df_final['compression_ok'] = df_final.apply(check_compression_row, axis=1)
    df_final = df_final[(df_final.lang_id_ok==True) & (df_final.compression_ok==True)]
    print("Writing rows to file")
    with open(args.final_output_file, 'w') as outfile:
        for index, row in df_final.iterrows():
            entry = {'messages':[{'role':'user', 'content':row['translation'].strip()}]}
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Done! Processed {str(len(sample_ids))} samples. Saved {str(df_final.shape[0])} samples. Final output file written to {args.final_output_file}")
        outfile.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
