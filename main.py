import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import datetime
import pre_process
import nltk
nltk.data.path.append('nltk_data')
import transformers
transformers.logging.set_verbosity_error()
import summarize
import compute_rouge

from summarizer import bert
from transformers import *

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('bert_model')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('bert_model')
custom_model = AutoModelForSeq2SeqLM.from_pretrained('bert_model', config=custom_config)

# Clear Screen
clear = lambda: os.system('cls')
clear()

# Define variable to store text from all text files
input_string = ""
optimised = ""

# Main function of the application
def main():
    global input_string, optimised
    try:

        print('-'*96)
        print('\n\t\t\t\tText Summarization Tool\n')
        print('-'*96)
        files = input("Enter dataset path (Absolute path to folder): ")
        print('-'*96)

        # Get files from folder
        if os.path.exists(files):

            print('\nProcessing files in the directory\n')

            start_time, files_list = datetime.datetime.now(), os.listdir(files)
            for i in range(len(files_list)):
                path = files +'\\'+ files_list[i]

                # Process text files only
                if path[-4:] == '.txt':
                    with open(path, 'r') as file:
                        new_file = file.read()
                        input_string += '\n' + new_file
                        file.close()

            print('Started pre-processing')
            # Start process (Tokenize and pre-process for BERT)
            cleaned = pre_process.main(input_string)
            cleaned = nltk.sent_tokenize(cleaned)
            cleaned = summarize.main(cleaned)

            print('\nStarted BERT Technique')

            # Configure BERT for summarization
            print('	 Configuring BERT Model')
            configure = bert.BertSummarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

            # Start summarization via BERT (Kmeans clustering)
            # Cluster size will be automatically determined
            summary = configure(" ".join(cleaned), return_as_list=True)

            # Get the top 3 sentences
            final_summary = summarize.processor(summary, num_sentences=3)
            final_optimised = ' '.join(final_summary)
            print('\nText Summarization Successful')

            # Write the summarized data to file
            print('\nWriting summary to file')
            Full_path = 'output\\summary.txt'

            with open(Full_path, 'w') as file:
                for sentence in final_summary: file.write('{0}\n'.format(sentence.strip()))
                file.close()

            # Process summary
            print('\nProcess completed successfully\n')
            print('-'*96)
            end_time = datetime.datetime.now()
            total_time = end_time - start_time
            redundancy = 100.00 - round((len(final_optimised) / len(input_string)) * 100)
               
            print('-'*96)  
            print('\t\t\t\tText Summarization - Process Summary')
            print('-'*96)
            print('Similarity Algorithm  : Cosine Similarity')
            print('Clustering Algorithm  : Kmeans')
            print('BERT Model            : bart-large-cnn')
            print('Input Length          :', len(input_string), 'words')
            print('Summary Length        :', len(final_optimised), 'words')
            print('Redundancy            : {0}%'.format(redundancy))
            print('Total Elapsed Time    :', total_time)
            print('-'*96)  

            # Write Report to File
            File_path = 'output\\report.txt'
            with open(File_path, 'w') as file:
                file.write('-'*55)
                file.write('\nText Summarization - Process Summary\n')
                file.write('-'*55)
                file.write('\nSimilarity Algorithm  : {0}\n'.format("Cosine Similarity"))
                file.write('Clustering Algorithm  : {0}\n'.format("Kmeans"))
                file.write('BERT Model            : {0}\n'.format("bart-large-cnn"))
                file.write('Input Length          : {0} words\n'.format(len(input_string)))
                file.write('Summary Length        : {0} words\n'.format(len(final_optimised)))
                file.write('Redundancy            : {0}%'.format(redundancy))
                file.write('Total Elapsed Time    : {0}\n'.format(total_time))
                file.write('-'*55)
                file.close()

            # Compute Rouge Score
            compute_rouge.main()

    except Exception as e:
        print('\n' + str(e))

if __name__ == '__main__':
    main()