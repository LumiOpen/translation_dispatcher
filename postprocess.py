import os
import json
import random
import sys
import glob
import gzip
import fasttext
import pandas as pd
from argparse import ArgumentParser

#language identifier
from huggingface_hub import hf_hub_download
import fasttext

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")   
model_glotlid = fasttext.load_model(model_path)


TOKENS_TO_REMOVE = ["<|user|>", "END", "Käännä suomeksi" , "Translate into"]
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

# remove samples with these words (case-insensitive)
FORBIDDEN_WORDS = ["openai", "mistral", "chatgpt", "tulu", "deepseek"]

LANGUAGE_CODES = {
        'bul': ['bul'],
        'hrv': ['hrv'],
        'ces': ['ces'],
        'dan': ['dan'],
        'nld': ['nld'],
        'eng': ['eng'],
        'est': ['ekk','est'],
        'fin': ['fin'],
        'fra': ['fra'],
        'deu': ['deu'],
        'ell': ['ell'],
        'hun': ['hun'],
        'gle': ['gle'],
        'isl': ['isl'],
        'ita': ['ita'],
        'lav': ['lav', 'lvs'],
        'lit': ['lit'],
        'mlt': ['mlt'],
        'nor': ['nor'],
        'pol': ['pol'],
        'por': ['por'],
        'ron': ['ron'],
        'slk': ['slk'],
        'slv': ['slv'],
        'spa': ['spa'],
        'swe': ['swe']
}

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--translation_output_file', default="output.jsonl", type=str)
    ap.add_argument('--complete_preprocessed_file', default="output.jsonl", type=str)
    ap.add_argument('--final_output_file', default="output.jsonl", type=str)
    ap.add_argument('--dataset_type', default="sft", type=str, help="dataset types: sft or dpo")
    ap.add_argument('--target_lang', default="fin", type=str)
    ap.add_argument('--lang_thresh', default=0.9, type=float, help="threshold for language detection")
    ap.add_argument('--max_lines_to_load', default=5000000, type=int, help="load N lines at a time to prevent OOM")
    return ap

def detect_language(text:str):
    """Given a text, it returns the Glotlid prediction as NLLB language code, e.g., Latn-eng
    """
    lang_code, score = model_glotlid.predict(text.replace("\n", " "))
    # extract 639-2 lang code (three-letter code)
    three_lang_code = lang_code[0].replace("__label__","").replace("_Latn","")
    # map 639-2 to 639-1 code if available
    # two_letter_code = GLOT_LANG_DICT.get(three_lang_code, "ERROR")
    return three_lang_code, score

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
        return False
    valid_text = True
    if (len(translated_text) / len(orig_text)) > max_len_ratio:
        valid_text = False
    return valid_text

def extract_orig_sent_row(row):
    if pd.isnull(row['prompt']):
        return ""
    else:
        prompt = row['prompt']
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

def check_untranslated_text(text, target_lang, thresh):  
    detected_lang, score = detect_language(text)
    ret_val = detected_lang + " " + str(score)
    if detected_lang == target_lang and score >= thresh:
        return True
    else:
        return False
    
def check_untranslated_row(row, target_lang, thresh):  
    detected_lang, score = detect_language(row['translation'])
    ret_val = target_lang + " " + detected_lang + " " + str(score)
    # if score > thresh:
    #     print(f"Text: {row['translation']}")
    #     print(f"Detected language: {ret_val}")
    #     print("--------------------")
    if detected_lang in LANGUAGE_CODES[target_lang] and score >= thresh:
        return True
    else:
        return False
    
def check_untranslated_dpo_row(row):    
    lang_chosen, score = detect_language(row['chosen'][0]['content'])
    lang_rejected, score = detect_language(row['rejected'][0]['content'])
    langs_prompt = []
    for turn in row['prompt']:
        lang_prompt, score = detect_language(turn['content'])
        langs_prompt.append(lang_prompt)
    if 'eng' not in langs_prompt and lang_rejected != 'eng' and lang_chosen != 'eng':
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

def jsonl_batch_reader(filename, batch_size):
    with open(filename) as f:
        while True:
            lines = []
            try:
                for _ in range(batch_size):
                    lines.append(json.loads(next(f)))
            except StopIteration:
                if lines:
                    yield pd.DataFrame(lines)
                break
            if lines:
                yield pd.DataFrame(lines)
                
def main(argv):
    args = argparser().parse_args(argv[1:])
    print(f"target language: {args.target_lang.upper()} | threshold: {args.lang_thresh}")
    df_all = pd.read_json(args.complete_preprocessed_file, lines=True)
    df_translate = pd.read_json(args.translation_output_file, lines=True)
    if args.dataset_type == 'sft':
        print("Post-processing SFT data")
        df_merged = pd.merge(df_all, df_translate, on=['sample_id', 'line_id'], how='left')
        print("df_merged:", len(df_merged))
        df_merged['translation'] = df_merged.apply(lambda row: row['translation_y'] if pd.notnull(row['translation_y']) else row['translation_x'], axis=1)
        sample_ids = sorted(df_merged.sample_id.unique())
        # print("Combining translated lines")
        df_final = {
                    'translation':[],
                    'orig_text':[],
                    'sample_id': []
                }
        for i, sample_id in enumerate(sample_ids):
            df_sample = df_merged[df_merged.sample_id==sample_id]
            # print("sample id:",sample_id)
            # print(df_sample)
            translation = "\n".join(list(df_sample.translation))
            orig_sents = list(df_sample.apply(extract_orig_sent_row, axis=1))
            orig_text = "\n".join(orig_sents)
            df_final['translation'].append(translation.strip())
            df_final['orig_text'].append(orig_text.strip())
            df_final['sample_id'].append(sample_id)
        df_final = pd.DataFrame.from_dict(df_final)
        df_final['lang_id_ok'] = df_final.apply(lambda x: check_untranslated_row(x, args.target_lang, args.lang_thresh), axis=1)
        df_final['compression_ok'] = df_final.apply(check_compression_row, axis=1)
        df_final['length_ok'] = df_final.apply(check_length_row, axis=1)
        df_final = df_final[(df_final.lang_id_ok!=False) & (df_final.compression_ok==True) & (df_final.length_ok==True)]
        print("Writing SFT rows to file")
        with open(args.final_output_file, 'w') as outfile:
            for index, row in df_final.iterrows():
                entry = {'messages':[
                                    {'role':'user', 
                                      'content':row['translation'].strip(),
                                      'orig_text': row['orig_text'],}
                                    ],
                        'sample_id': row['sample_id']}
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
