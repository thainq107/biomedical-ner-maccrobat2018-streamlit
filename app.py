import streamlit as st
from transformers import pipeline

classifier = pipeline(
    "token-classification",
    model="thainq107/ner-biomedical-maccrobat2018",
    aggregation_strategy="simple"
)

def main():
  st.title('Biomedical NER')
  st.title('Model: DistilBERT. Dataset: MACCROBAT2018')
  sentence = """A 48 year - old female presented with vaginal bleeding and abnormal Pap smears .
  Upon diagnosis of invasive non - keratinizing SCC of the cervix ,
  she underwent a radical hysterectomy with salpingo - oophorectomy
  which demonstrated positive spread to the pelvic lymph nodes and the parametrium .
  Pathological examination revealed that the tumour also extensively involved the lower uterine segment .
  """
  text_input = st.text_input("Sentence: ", sentence)
  results = classifier(text_input)
  for result in results:
      entity = result["entity_group"]
      word = result["word"]
      st.success(f'Word: {word} - Entity {entity}') 

if __name__ == '__main__':
     main() 
