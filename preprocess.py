import os
import json
import random
import sys
import glob
import pandas as pd
from argparse import ArgumentParser

# Prompt templates
USER_ASSISTANT_TEMPLATE = '<|user|>{src_sent}\n<|assistant|>{trg_sent}\n{end}'
DOUBLE_HASH_TEMPLATE = "## Translate into {trg_lang_name}: {src_sent}\n{trg_sent}"

# ICL examples per language (samples from FLORES-101 and Tatoeba dev sets)
FLORES_SENT_INDICES = [162, 678, 850, 898, 674, 724, 83, 351]
FLORES_PATH = "/scratch/project_462000353/finetuning_data/FLORES-200"

# Language dictionary
TARGET_LANGUAGE_NAMES = {
        'bul': 'Bulgarian',
        'hrv': 'Croatian',
        'ces': 'Czech',
        'dan': 'Danish',
        'nld': 'Dutch',
        'eng': 'English',
        'est': 'Estonian',
        'fin': 'Finnish',
        'fra': 'French',
        'deu': 'German',
        'ell': 'Greek',
        'hun': 'Hungarian',
        'gle': 'Irish',
        'isl': 'Icelandic',
        'ita': 'Italian',
        'lav': 'Latvian',
        'lit': 'Lithuanian',
        'mlt': 'Maltese',
        'nor': 'Norwegian',
        'pol': 'Polish',
        'por': 'Portuguese',
        'ron': 'Romanian',
        'slk': 'Slovak',
        'slv': 'Slovenian',
        'spa': 'Spanish',
        'swe': 'Swedish'
}

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--input_file', default="input.jsonl", type=str)
    ap.add_argument('--translation_input_file', default="output.jsonl", type=str)
    ap.add_argument('--preprocessed_file', default="output.jsonl", type=str)
    ap.add_argument('--trg_lang', default=None, type=str, help="3-letter language code")
    ap.add_argument('--prompt_format', default="double_hash", type=str, help="double_hash or user_assistant")
    return ap

def format_prompt_user_assistant_template(src_sents, trg_sents):
    template = USER_ASSISTANT_TEMPLATE
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(
                        src_sent=src,
                        trg_sent=trg,
                        end="END"
                )
        prompts.append(text) 
    few_shot_prompt = "\n\n".join(prompts)
    return few_shot_prompt

def format_prompt_double_hash_template(trg_lang, src_sents, trg_sents):
    template = DOUBLE_HASH_TEMPLATE
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(
                        trg_lang_name=TARGET_LANGUAGE_NAMES[trg_lang],
                        src_sent=src,
                        trg_sent=trg
                )
        prompts.append(text) 
    few_shot_prompt = "\n\n".join(prompts)
    return few_shot_prompt

def create_few_shot_prompt(trg_lang, prompt_format):
        src_lang = "eng"
        flores_src_sentences = open(os.path.join(FLORES_PATH, src_lang+"-dev.txt")).readlines()
        flores_trg_sentences = open(os.path.join(FLORES_PATH, trg_lang+"-dev.txt")).readlines()
        src_sents = [flores_src_sentences[sent_index].strip() for sent_index in FLORES_SENT_INDICES]
        trg_sents = [flores_trg_sentences[sent_index].strip() for sent_index in FLORES_SENT_INDICES]
        if prompt_format == 'double_hash':
            prompt = format_prompt_double_hash_template(trg_lang, src_sents, trg_sents)
        else:
            prompt = format_prompt_user_assistant_template(src_sents, trg_sents)
        return prompt

def prepare_content_for_translation(content, sample_id, trg_lang, few_shot_prompt, prompt_format):
    content_to_translate = []
    content_all = []
    line_id = 1 # start line_id at 1
    is_code = False
    # print(f"\n---------SAMPLE ID: {sample_id}---------\n")
    # print(f"\nCONTENT:\n{content}\n")
    paragraphs_double_newline = content.split("\n\n")
    for paragraph_double in paragraphs_double_newline:
        if len(paragraph_double) > 0:
            paragraphs_single_newline = paragraph_double.split("\n")
            for paragraph in paragraphs_single_newline:
                if len(paragraph) > 0:
                    # print(f"\n-----LINE ID: {line_id}-----\n")
                    if "```" in paragraph and is_code is False:
                        is_code = True
                    elif "```" in paragraph and is_code is True:
                        is_code = False
                    if is_code is True:
                        if "```" in paragraph:
                            paragraph = "\n\n" + paragraph.strip()
                        content_all.append({
                                                'content':paragraph, 
                                                'translate':False,
                                                'sample_id': sample_id,
                                                'line_id': line_id,
                                                'translation': paragraph
                                            })
                        line_id += 1
                    else:  
                        if paragraph == "```":
                            content_all.append({
                                                    'content':paragraph, 
                                                    'translate':False,
                                                    'sample_id': sample_id,
                                                    'line_id': line_id,
                                                    'translation': paragraph
                                                })
                            line_id += 1
                        else:
                            # Format paragraph and prepend few-shot prompt
                            if prompt_format == "double_hash":
                                formatted_para = DOUBLE_HASH_TEMPLATE.format(
                                    trg_lang_name=TARGET_LANGUAGE_NAMES[trg_lang],
                                    src_sent=paragraph.strip(),
                                    trg_sent="" 
                                )
                            else:
                                formatted_para = USER_ASSISTANT_TEMPLATE.format(
                                    src_sent=paragraph.strip(),
                                    trg_sent="",
                                    end=""
                                )
                            formatted_prompt = few_shot_prompt + "\n\n" + formatted_para.strip()
                            # print(f"\nFINAL PROMPT:\n{formatted_prompt}")
                            content_all.append({
                                                'content':formatted_prompt, 
                                                'translate':True,
                                                'sample_id': sample_id,
                                                'line_id': line_id,
                                                'translation':''
                                                })
                            content_to_translate.append({
                                                        'content':formatted_prompt, 
                                                        'translate':True,
                                                        'sample_id': sample_id,
                                                        'line_id': line_id,
                                                        'translation':''
                                                        })
                            line_id += 1
    return content_to_translate, content_all

def main(argv):
    args = argparser().parse_args(argv[1:])
    file = open(args.input_file)
    few_shot_prompt = create_few_shot_prompt(args.trg_lang, args.prompt_format)
    # print(f"\nFEW-SHOT PROMPT:\n{few_shot_prompt}\n")
    outfile = open(args.preprocessed_file, 'w')
    translation_outfile = open(args.translation_input_file, 'w')
    for i, line in enumerate(file):
        entry = json.loads(line)
        content_to_translate, content_all = prepare_content_for_translation(content=entry['messages'][0]['content'], 
                                                                sample_id=i+1, # start sample_id at 1 
                                                                trg_lang=args.trg_lang,
                                                                few_shot_prompt=few_shot_prompt,
                                                                prompt_format=args.prompt_format)
        for content_dict in content_all:
            outfile.write(json.dumps(content_dict, ensure_ascii=False) + "\n")
        for content_dict in content_to_translate:
            translation_outfile.write(json.dumps(content_dict, ensure_ascii=False) + "\n")

    


if __name__ == '__main__':
    sys.exit(main(sys.argv))
