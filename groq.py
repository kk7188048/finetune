from datasets import load_dataset

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

dataset = load_dataset('csv', data_files='customers-100.csv')

question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")