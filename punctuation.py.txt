from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import re

class PunctuationModel():
    def __init__(self, model_name="prithivida/psst-punctuation-multilingual", device=0, chunk_size=100, overlap=20):
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, grouped_entities=False, device=device)

    def preprocess(self, text):
        # Remove markers except for markers in numbers 
        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)","",text) 
        # TODO: match acronyms https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        text = text.split()
        return text

    def restore_punctuation(self, text):
        result = self.predict(self.preprocess(text))
        return self.prediction_to_text(result)
        
    def overlap_chunks(self, lst):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), self.chunk_size - self.overlap):
            yield lst[i:i + self.chunk_size]

    def predict(self, words):
        batches = list(self.overlap_chunks(words))

        # If the last batch is smaller than the overlap, 
        # we can just remove it
        if len(batches[-1]) <= self.overlap:
            batches.pop()

        tagged_words = []     
        for batch in batches:
            # Use last batch completely
            if batch == batches[-1]: 
                self.overlap = 0

            text = " ".join(batch)
            result = self.pipeline(text)
                
            char_index = 0
            result_index = 0
            for word in batch[:len(batch)-self.overlap]:                
                char_index += len(word) + 1
                # If any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end        
                label = "0"
                while result_index < len(result) and char_index > result[result_index]["end"] :
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1                        
                tagged_words.append([word,label, score])
        
        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self,prediction):
        result = ""
        for word, label, _ in prediction:
            result += word
            if label == "0":
                result += " "
            if label in ".,?-:":
                result += label+" "
        return result.strip()

if __name__ == "__main__":    
    model = PunctuationModel(model_name="prithivida/psst-punctuation-multilingual", device=0, chunk_size=100, overlap=20)

    text = "this is a sample sentence that has no punctuation but we want to add it back"
    # Restore missing punctuation
    result = model.restore_punctuation(text)
    print(result)

    clean_text = model.preprocess(text)
    labled_words = model.predict(clean_text)
    print(labled_words)
