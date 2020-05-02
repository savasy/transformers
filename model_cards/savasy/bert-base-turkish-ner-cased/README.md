Ner Model For Turkish Language, 





from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner=pipeline('ner', model=model, tokenizer=tokenizer)
ner("Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.")
