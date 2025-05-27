import os
import json
import random
import sys
import glob
import gzip
import fasttext
import pandas as pd
from argparse import ArgumentParser

FASTTEXT_LID_BINARY = "/scratch/project_462000353/zosaelai2/models/lid.176.bin"
fasttext.FastText.eprint = lambda x: None
lid_model = fasttext.load_model(FASTTEXT_LID_BINARY)

TOKENS_TO_REMOVE = ["<|user|>", "END", "Käännä suomeksi" , "Translate into"]
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

# remove samples with these words (case-insensitive)
FORBIDDEN_WORDS = ["openai", "mistral", "chatgpt", "tulu", "deepseek"]

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--translation_output_file', default="output.jsonl", type=str)
    ap.add_argument('--complete_preprocessed_file', default="output.jsonl", type=str)
    ap.add_argument('--final_output_file', default="output.jsonl", type=str)
    ap.add_argument('--dataset_type', default="sft", type=str, help="dataset types: sft or dpo")
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

def get_translation_length_ratio(translated_text, orig_text, max_len_ratio=1.5):
    if len(translated_text) == 0 or len(orig_text) == 0:
        # print("translated_text:", translated_text)
        # print("orig_text:", orig_text)
        # print("-----")
        return False
    valid_text = True
    if (len(translated_text) / len(orig_text)) > max_len_ratio:
        valid_text = False
    return valid_text

def extract_orig_sent_row(row):
    prompt = row['content']
    orig_sent = prompt[prompt.rfind(USER_TOKEN)+len(USER_TOKEN):prompt.rfind(ASSISTANT_TOKEN)].strip()
    return orig_sent


def check_turns(row):
    num_turns = len(row['messages'])/2
    return num_turns

def check_length_row(row):
    length_ok = get_translation_length_ratio(row['translation'], row['orig_text'])
    return length_ok

def check_length_dpo_row(row):
    length_chosen_ok = get_translation_length_ratio(row['chosen'][0]['content'], row['chosen'][0]['orig_text'])
    length_rejected_ok = get_translation_length_ratio(row['rejected'][0]['content'], row['rejected'][0]['orig_text'])
    length_prompts_ok = []
    for turn in row['prompt']:
        length_ok = get_translation_length_ratio(turn['content'], turn['orig_text'])
        length_prompts_ok.append(length_ok)
    return all([length_chosen_ok, length_rejected_ok] + length_prompts_ok)

def check_untranslated_row(row):    
    detected_lang, prob = detect_language(row['translation'])
    if detected_lang == 'en':
        return False
    else:
        return True
    
def check_untranslated_dpo_row(row):    
    lang_chosen, prob = detect_language(row['chosen'][0]['content'])
    lang_rejected, prob = detect_language(row['rejected'][0]['content'])
    langs_prompt = []
    for turn in row['prompt']:
        lang_prompt, prob = detect_language(turn['content'])
        langs_prompt.append(lang_prompt)
    if 'en' not in langs_prompt and lang_rejected != 'en' and lang_chosen != 'en':
        return True
    else:
        return False

def check_compression_row(row):    
    compression_ok = check_compression(row['translation'])
    return compression_ok

def check_compression_dpo_row(row):    
    compression_chosen_ok = check_compression(row['chosen'][0]['content'])
    compression_rejected_ok = check_compression(row['chosen'][0]['content'])
    compression_prompts_ok = []
    for turn in row['prompt']:
        compression_ok = check_compression(turn['content'])
        compression_prompts_ok.append(compression_ok)
    return all([compression_chosen_ok, compression_rejected_ok] + compression_prompts_ok)

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
    # extract orig text from preproc data
    df_all['orig_sent'] = df_all.apply(extract_orig_sent_row, axis=1)
    if args.dataset_type == 'sft':
        print("Post-processing SFT data")
        df_merged = pd.merge(df_all, df_translate, on=['sample_id', 'line_id'], how='left')
        df_merged['translation'] = df_merged.apply(lambda row: row['translation_y'] if pd.notnull(row['translation_y']) else row['translation_x'], axis=1)
        sample_ids = sorted(df_merged.sample_id.unique())
        # print("Combining translated lines")
        df_final = {'translation':[]}
        for i, sample_id in enumerate(sample_ids):
            translation = "\n".join(list(df_merged[df_merged.sample_id==sample_id].translation))
            orig_text = "\n".join(list(df_merged[df_merged.sample_id==sample_id].orig_sent))
            df_final['translation'].append(translation.strip())
            df_final['orig_text'].append(orig_text.strip())
        df_final = pd.DataFrame.from_dict(df_final)
        # print("Checking lang_id and compression")
        df_final['lang_id_ok'] = df_final.apply(check_untranslated_row, axis=1)
        df_final['compression_ok'] = df_final.apply(check_compression_row, axis=1)
        df_final['length_ok'] = df_final.apply(check_length_row, axis=1)
        df_final = df_final[(df_final.lang_id_ok==True) & (df_final.compression_ok==True) & (df_final.length_ok==True)]
        print("Writing SFT rows to file")
        with open(args.final_output_file, 'w') as outfile:
            for index, row in df_final.iterrows():
                entry = {'messages':[{'role':'user', 'content':row['translation'].strip()}]}
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Done! Processed {str(len(sample_ids))} samples. Saved {str(df_final.shape[0])} samples. Final output file written to {args.final_output_file}")
            outfile.close()
    else:
        print("Post-processing DPO data")
        df_merged = pd.merge(df_all, df_translate, on=['sample_id', 'column', 'turn_id', 'role', 'line_id'], how='left')
        df_merged['translation'] = df_merged.apply(lambda row: row['translation_y'] if pd.notnull(row['translation_y']) else row['translation_x'], axis=1)
        sample_ids = sorted(df_merged.sample_id.unique())
        columns = ['prompt', 'chosen', 'rejected']
        roles = ['user', 'assistant']
        df_final = {'prompt':[], 'chosen':[], 'rejected':[]}
        print(f"Post-processing {int(len(sample_ids))} samples")
        for i, sample_id in enumerate(sample_ids):
            # df_entry = {'prompt':[], 'chosen':[], 'rejected':[]}
            sample = df_merged[df_merged.sample_id==sample_id]
            for col_name in columns:
                if col_name != 'prompt':
                    # chosen and rejected columns have a single turn
                    col_df = sample[sample.column==col_name]
                    translation = "\n".join(list(col_df.translation))
                    orig_text  = "\n".join(list(col_df.orig_sent))
                    # df_entry[col_name].append({'role':'assistant', 
                    #                         'content':translation.strip(), 
                    #                         'orig_content':orig_text.strip()})
                    df_final[col_name].append([{'role':'assistant',
                                               'content':translation.strip(),
                                               'orig_text':orig_text.strip()
                                               }])
                else:
                    # prompt are multi-turn (each turn has a user and assistant roles)
                    final_prompt = []
                    prompt_df = sample[sample.column=='prompt']
                    turn_ids = sorted(prompt_df.turn_id.unique())
                    for turn_id in turn_ids:
                        turn_df = prompt_df[prompt_df.turn_id==turn_id]
                        roles_in_turns = turn_df.role.unique()
                        for role in roles_in_turns:
                            role_df = turn_df[turn_df.role==role]
                            translation = "\n".join(list(role_df.translation))
                            orig_text  = "\n".join(list(role_df.orig_sent))
                            # df_entry['prompt'].append({'role':role, 
                            #                            'content':translation.strip(), 
                            #                            'orig_content':orig_text.strip()})
                            final_prompt.append({'role':role,
                                                 'content':translation.strip(),
                                                 'orig_text':orig_text.strip()
                                                 })
                    df_final['prompt'].append(final_prompt)
        df_final = pd.DataFrame.from_dict(df_final)
        # df_final.to_json(args.final_output_file,  orient="records", lines=True, force_ascii=False)
        df_final['lang_id_ok'] = df_final.apply(check_untranslated_dpo_row, axis=1)
        df_final['compression_ok'] = df_final.apply(check_compression_dpo_row, axis=1)
        df_final['length_ok'] = df_final.apply(check_length_dpo_row, axis=1)
        df_final = df_final[(df_final.lang_id_ok==True) & (df_final.compression_ok==True) & (df_final.length_ok==True)]
        print("Writing DPO rows to file")
        with open(args.final_output_file, 'w') as outfile:
            for index, row in df_final.iterrows():
                # entry = {'messages':[{'role':'user', 'content':row['translation'].strip()}]}
                entry = {'prompt': [{'role':msg['role'], 'content': msg['content']} for msg in row['prompt']],
                         'chosen': [{'role':msg['role'], 'content': msg['content']} for msg in row['chosen']],
                         'rejected': [{'role':msg['role'], 'content': msg['content']} for msg in row['rejected']]
                        }
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Done! Processed {str(len(sample_ids))} samples. Saved {str(df_final.shape[0])} samples. Final output file written to {args.final_output_file}")
            outfile.close()
        
        




if __name__ == '__main__':
    sys.exit(main(sys.argv))
